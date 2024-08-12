//! # wgsl_to_wgpu
//! wgsl_to_wgpu is a library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Getting Started
//! The [create_shader_module] function is intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! ```rust no_run
//! // build.rs
//! use wgsl_to_wgpu::{create_shader_module, MatrixVectorTypes, WriteOptions};
//!
//! fn main() {
//!     println!("cargo:rerun-if-changed=src/shader.wgsl");
//!
//!     // Read the shader source file.
//!     let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();
//!
//!     // Configure the output based on the dependencies for the project.
//!     let options = WriteOptions {
//!         derive_bytemuck_vertex: true,
//!         derive_encase_host_shareable: true,
//!         matrix_vector_types: MatrixVectorTypes::Glam,
//!         ..Default::default()
//!     };
//!
//!     // Generate the bindings.
//!     let text = create_shader_module(&wgsl_source, "shader.wgsl", options).unwrap();
//!     std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
//! }
//! ```

extern crate wgpu_types as wgpu;

use bindgroup::{bind_groups_module, get_bind_group_data};
use consts::pipeline_overridable_constants;
use entry::{entry_point_constants, fragment_states, vertex_states, vertex_struct_methods};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Index};
use thiserror::Error;

mod bindgroup;
mod consts;
mod entry;
mod structs;
mod wgsl;

/// Errors while generating Rust source for a WGSl shader module.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum CreateModuleError {
    /// Bind group sets must be consecutive and start from 0.
    /// See `bind_group_layouts` for
    /// [PipelineLayoutDescriptor](https://docs.rs/wgpu/latest/wgpu/struct.PipelineLayoutDescriptor.html#).
    #[error("bind groups are non-consecutive or do not start from 0")]
    NonConsecutiveBindGroups,

    /// Each binding resource must be associated with exactly one binding index.
    #[error("duplicate binding found with index `{binding}`")]
    DuplicateBinding { binding: u32 },
}

/// Options for configuring the generated bindings to work with additional dependencies.
/// Use [WriteOptions::default] for only requiring WGPU itself.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct WriteOptions {
    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for WGSL vertex input structs when `true`.
    pub derive_bytemuck_vertex: bool,

    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for user defined WGSL structs for host-shareable types (uniform and storage buffers) when `true`.
    ///
    /// This will generate compile time assertions to check that the memory layout
    /// of structs and struct fields matches what is expected by WGSL.
    /// This does not account for all layout and alignment rules like storage buffer offset alignment.
    ///
    /// Most applications should instead handle these requirements more reliably at runtime using encase.
    pub derive_bytemuck_host_shareable: bool,

    /// Derive [encase::ShaderType](https://docs.rs/encase/latest/encase/trait.ShaderType.html#)
    /// for user defined WGSL structs for host-shareable types (uniform and storage buffers) when `true`.
    /// Use [MatrixVectorTypes::Glam] for best compatibility.
    pub derive_encase_host_shareable: bool,

    /// Derive [serde::Serialize](https://docs.rs/serde/1.0.159/serde/trait.Serialize.html)
    /// and [serde::Deserialize](https://docs.rs/serde/1.0.159/serde/trait.Deserialize.html)
    /// for user defined WGSL structs when `true`.
    pub derive_serde: bool,

    /// The format to use for matrix and vector types.
    pub matrix_vector_types: MatrixVectorTypes,
}

/// The format to use for matrix and vector types.
/// Note that the generated types for the same WGSL type may differ in size or alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixVectorTypes {
    /// Rust types like `[f32; 4]` or `[[f32; 4]; 4]`.
    Rust,

    /// `glam` types like `glam::Vec4` or `glam::Mat4`.
    /// Types not representable by `glam` like `mat2x3<f32>` will use the output from [MatrixVectorTypes::Rust].
    Glam,

    /// `nalgebra` types like `nalgebra::SVector<f64, 4>` or `nalgebra::SMatrix<f32, 2, 3>`.
    Nalgebra,
}

impl Default for MatrixVectorTypes {
    fn default() -> Self {
        Self::Rust
    }
}

/// Generates a Rust module for a WGSL shader included via [include_str].
///
/// The `wgsl_include_path` should be a valid input to [include_str] in the generated file's location.
/// The included contents should be identical to `wgsl_source`.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/**
```rust no_run
// build.rs
fn main() {
    println!("cargo:rerun-if-changed=src/shader.wgsl");

    // Read the shader source file.
    let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();

    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions {
        derive_bytemuck_vertex: true,
        derive_encase_host_shareable: true,
        matrix_vector_types: wgsl_to_wgpu::MatrixVectorTypes::Glam,
        ..Default::default()
    };

    // Generate the bindings.
    let text = wgsl_to_wgpu::create_shader_module(&wgsl_source, "shader.wgsl", options).unwrap();
    std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
}
```
 */
pub fn create_shader_module(
    wgsl_source: &str,
    wgsl_include_path: &str,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    create_shader_module_inner(wgsl_source, Some(wgsl_include_path), options)
}

// TODO: Show how to convert a naga module back to wgsl.
/// Generates a Rust module for a WGSL shader embedded as a string literal.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/// The source string does not need to be from an actual file on disk.
/// This allows applying build time operations like preprocessor defines.
/**
```rust no_run
// build.rs
# fn generate_wgsl_source_string() -> String { String::new() }
fn main() {
    // Generate the shader at build time.
    let wgsl_source = generate_wgsl_source_string();

    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions {
        derive_bytemuck_vertex: true,
        derive_encase_host_shareable: true,
        matrix_vector_types: wgsl_to_wgpu::MatrixVectorTypes::Glam,
        ..Default::default()
    };

    // Generate the bindings.
    let text = wgsl_to_wgpu::create_shader_module_embedded(&wgsl_source, options).unwrap();
    std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
}
```
 */
pub fn create_shader_module_embedded(
    wgsl_source: &str,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    create_shader_module_inner(wgsl_source, None, options)
}

fn create_shader_module_inner(
    wgsl_source: &str,
    wgsl_include_path: Option<&str>,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap();

    let bind_group_data = get_bind_group_data(&module)?;
    let shader_stages = wgsl::shader_stages(&module);

    // Write all the structs, including uniforms and entry function inputs.
    let structs = structs::structs(&module, options);
    let consts = consts::consts(&module);
    let bind_groups_module = bind_groups_module(&bind_group_data, shader_stages);
    let vertex_module = vertex_struct_methods(&module);
    let compute_module = compute_module(&module);
    let entry_point_constants = entry_point_constants(&module);
    let vertex_states = vertex_states(&module);
    let fragment_states = fragment_states(&module);

    // Use a string literal if no include path is provided.
    let included_source = wgsl_include_path
        .map(|p| quote!(include_str!(#p)))
        .unwrap_or_else(|| quote!(#wgsl_source));

    let create_shader_module = quote! {
        pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
            let source = std::borrow::Cow::Borrowed(#included_source);
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source)
            })
        }
    };

    let bind_group_layouts: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("BindGroup", *group_no);
            quote!(bind_groups::#group::get_bind_group_layout(device))
        })
        .collect();

    let push_constant_range = push_constant_range(&module, shader_stages);

    let create_pipeline_layout = quote! {
        pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    #(&#bind_group_layouts),*
                ],
                push_constant_ranges: &[#push_constant_range],
            })
        }
    };

    let override_constants = pipeline_overridable_constants(&module);

    let output = quote! {
        #(#structs)*
        #(#consts)*
        #override_constants
        #bind_groups_module
        #vertex_module
        #compute_module
        #entry_point_constants
        #vertex_states
        #fragment_states
        #create_shader_module
        #create_pipeline_layout
    };
    Ok(pretty_print(&output))
}

fn push_constant_range(
    module: &naga::Module,
    shader_stages: wgpu::ShaderStages,
) -> Option<TokenStream> {
    // Assume only one variable is used with var<push_constant> in WGSL.
    let push_constant_size = module.global_variables.iter().find_map(|g| {
        if g.1.space == naga::AddressSpace::PushConstant {
            Some(module.types[g.1.ty].inner.size(module.to_ctx()))
        } else {
            None
        }
    });

    let stages = quote_shader_stages(shader_stages);

    // Use a single push constant range for all shader stages.
    // This allows easily setting push constants in a single call with offset 0.
    push_constant_size.map(|size| {
        let size = Index::from(size as usize);
        quote! {
            wgpu::PushConstantRange {
                stages: #stages,
                range: 0..#size
            }
        }
    })
}

fn pretty_print(tokens: &TokenStream) -> String {
    let file = syn::parse_file(&tokens.to_string()).unwrap();
    prettyplease::unparse(&file)
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{name}{index}"), Span::call_site())
}

fn name_to_ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

fn compute_module(module: &naga::Module) -> TokenStream {
    let entry_points: Vec<_> = module
        .entry_points
        .iter()
        .filter_map(|e| {
            if e.stage == naga::ShaderStage::Compute {
                let workgroup_size_constant = workgroup_size(e);
                let create_pipeline = create_compute_pipeline(e);

                Some(quote! {
                    #workgroup_size_constant
                    #create_pipeline
                })
            } else {
                None
            }
        })
        .collect();

    if entry_points.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod compute {
                #(#entry_points)*
            }
        }
    }
}

fn create_compute_pipeline(e: &naga::EntryPoint) -> TokenStream {
    // Compute pipeline creation has few parameters and can be generated.
    let pipeline_name = Ident::new(&format!("create_{}_pipeline", e.name), Span::call_site());
    let entry_point = &e.name;
    // TODO: Include a user supplied module name in the label?
    let label = format!("Compute Pipeline {}", e.name);
    quote! {
        pub fn #pipeline_name(device: &wgpu::Device) -> wgpu::ComputePipeline {
            let module = super::create_shader_module(device);
            let layout = super::create_pipeline_layout(device);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(#label),
                layout: Some(&layout),
                module: &module,
                entry_point: #entry_point,
                compilation_options: Default::default(),
                cache: Default::default(),
            })
        }
    }
}

fn workgroup_size(e: &naga::EntryPoint) -> TokenStream {
    // Use Index to avoid specifying the type on literals.
    let name = Ident::new(
        &format!("{}_WORKGROUP_SIZE", e.name.to_uppercase()),
        Span::call_site(),
    );
    let [x, y, z] = e.workgroup_size.map(|s| Index::from(s as usize));
    quote!(pub const #name: [u32; 3] = [#x, #y, #z];)
}

fn quote_shader_stages(shader_stages: wgpu::ShaderStages) -> TokenStream {
    match shader_stages {
        wgpu::ShaderStages::VERTEX_FRAGMENT => quote!(wgpu::ShaderStages::VERTEX_FRAGMENT),
        wgpu::ShaderStages::COMPUTE => quote!(wgpu::ShaderStages::COMPUTE),
        wgpu::ShaderStages::VERTEX => quote!(wgpu::ShaderStages::VERTEX),
        wgpu::ShaderStages::FRAGMENT => quote!(wgpu::ShaderStages::FRAGMENT),
        _ => todo!(),
    }
}

// Tokenstreams can't be compared directly using PartialEq.
// Use pretty_print to normalize the formatting and compare strings.
// Use a colored diff output to make differences easier to see.
#[cfg(test)]
#[macro_export]
macro_rules! assert_tokens_eq {
    ($a:expr, $b:expr) => {
        pretty_assertions::assert_eq!(crate::pretty_print(&$a), crate::pretty_print(&$b))
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use indoc::indoc;

    #[test]
    fn create_shader_module_include_source() {
        let source = indoc! {r#"
            var<push_constant> consts: vec4<f32>;

            @fragment
            fn fs_main() {}
        "#};

        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();

        pretty_assertions::assert_eq!(
            indoc! {r#"
                pub const ENTRY_FS_MAIN: &str = "fs_main";
                #[derive(Debug)]
                pub struct FragmentEntry<const N: usize> {
                    pub entry_point: &'static str,
                    pub targets: [Option<wgpu::ColorTargetState>; N],
                    pub constants: std::collections::HashMap<String, f64>,
                }
                pub fn fragment_state<'a, const N: usize>(
                    module: &'a wgpu::ShaderModule,
                    entry: &'a FragmentEntry<N>,
                ) -> wgpu::FragmentState<'a> {
                    wgpu::FragmentState {
                        module,
                        entry_point: entry.entry_point,
                        targets: &entry.targets,
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &entry.constants,
                            ..Default::default()
                        },
                    }
                }
                pub fn fs_main_entry(targets: [Option<wgpu::ColorTargetState>; 0]) -> FragmentEntry<0> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_MAIN,
                        targets,
                        constants: Default::default(),
                    }
                }
                pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                    let source = std::borrow::Cow::Borrowed(include_str!("shader.wgsl"));
                    device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Wgsl(source),
                        })
                }
                pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
                    device
                        .create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[],
                                push_constant_ranges: &[
                                    wgpu::PushConstantRange {
                                        stages: wgpu::ShaderStages::FRAGMENT,
                                        range: 0..16,
                                    },
                                ],
                            },
                        )
                }
            "#},
            actual
        );
    }

    #[test]
    fn create_shader_module_embed_source() {
        let source = indoc! {r#"
            @fragment
            fn fs_main() {}
        "#};

        let actual = create_shader_module_embedded(source, WriteOptions::default()).unwrap();

        pretty_assertions::assert_eq!(
            indoc! {r#"
                pub const ENTRY_FS_MAIN: &str = "fs_main";
                #[derive(Debug)]
                pub struct FragmentEntry<const N: usize> {
                    pub entry_point: &'static str,
                    pub targets: [Option<wgpu::ColorTargetState>; N],
                    pub constants: std::collections::HashMap<String, f64>,
                }
                pub fn fragment_state<'a, const N: usize>(
                    module: &'a wgpu::ShaderModule,
                    entry: &'a FragmentEntry<N>,
                ) -> wgpu::FragmentState<'a> {
                    wgpu::FragmentState {
                        module,
                        entry_point: entry.entry_point,
                        targets: &entry.targets,
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &entry.constants,
                            ..Default::default()
                        },
                    }
                }
                pub fn fs_main_entry(targets: [Option<wgpu::ColorTargetState>; 0]) -> FragmentEntry<0> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_MAIN,
                        targets,
                        constants: Default::default(),
                    }
                }
                pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                    let source = std::borrow::Cow::Borrowed("@fragment\nfn fs_main() {}\n");
                    device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Wgsl(source),
                        })
                }
                pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
                    device
                        .create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[],
                                push_constant_ranges: &[],
                            },
                        )
                }
            "#},
            actual
        );
    }

    #[test]
    fn create_shader_module_consecutive_bind_groups() {
        let source = indoc! {r#"
            struct A {
                f: vec4<f32>
            };
            @group(0) @binding(0) var<uniform> a: A;
            @group(1) @binding(0) var<uniform> b: A;

            @vertex
            fn vs_main() {}

            @fragment
            fn fs_main() {}
        "#};

        create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
    }

    #[test]
    fn create_shader_module_non_consecutive_bind_groups() {
        let source = indoc! {r#"
            @group(0) @binding(0) var<uniform> a: vec4<f32>;
            @group(1) @binding(0) var<uniform> b: vec4<f32>;
            @group(3) @binding(0) var<uniform> c: vec4<f32>;

            @fragment
            fn main() {}
        "#};

        let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());
        assert!(matches!(
            result,
            Err(CreateModuleError::NonConsecutiveBindGroups)
        ));
    }

    #[test]
    fn create_shader_module_repeated_bindings() {
        let source = indoc! {r#"
            struct A {
                f: vec4<f32>
            };
            @group(0) @binding(2) var<uniform> a: A;
            @group(0) @binding(2) var<uniform> b: A;

            @fragment
            fn main() {}
        "#};

        let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());
        assert!(matches!(
            result,
            Err(CreateModuleError::DuplicateBinding { binding: 2 })
        ));
    }

    #[test]
    fn write_vertex_module_empty() {
        let source = indoc! {r#"
            @vertex
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module);

        assert_tokens_eq!(quote!(), actual);
    }

    #[test]
    fn write_vertex_module_single_input_float32() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: f32,
                @location(1) b: vec2<f32>,
                @location(2) c: vec3<f32>,
                @location(3) d: vec4<f32>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_float64() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: f64,
                @location(1) b: vec2<f64>,
                @location(2) c: vec3<f64>,
                @location(3) d: vec4<f64>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_sint32() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: i32,
                @location(1) a: vec2<i32>,
                @location(2) a: vec3<i32>,
                @location(3) a: vec4<i32>,

            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32x2,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32x3,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32x4,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_uint32() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: u32,
                @location(1) b: vec2<u32>,
                @location(2) c: vec3<u32>,
                @location(3) d: vec4<u32>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            actual
        );
    }

    #[test]
    fn write_compute_module_empty() {
        let source = indoc! {r#"
            @vertex
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = compute_module(&module);

        assert_tokens_eq!(quote!(), actual);
    }

    #[test]
    fn write_compute_module_multiple_entries() {
        let source = indoc! {r#"
            @compute
            @workgroup_size(1,2,3)
            fn main1() {}

            @compute
            @workgroup_size(256)
            fn main2() {}
        "#
        };

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = compute_module(&module);

        assert_tokens_eq!(
            quote! {
                pub mod compute {
                    pub const MAIN1_WORKGROUP_SIZE: [u32; 3] = [1, 2, 3];
                    pub fn create_main1_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
                        let module = super::create_shader_module(device);
                        let layout = super::create_pipeline_layout(device);
                        device
                            .create_compute_pipeline(
                                &wgpu::ComputePipelineDescriptor {
                                    label: Some("Compute Pipeline main1"),
                                    layout: Some(&layout),
                                    module: &module,
                                    entry_point: "main1",
                                    compilation_options: Default::default(),
                                    cache: Default::default(),
                                },
                            )
                    }
                    pub const MAIN2_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
                    pub fn create_main2_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
                        let module = super::create_shader_module(device);
                        let layout = super::create_pipeline_layout(device);
                        device
                            .create_compute_pipeline(
                                &wgpu::ComputePipelineDescriptor {
                                    label: Some("Compute Pipeline main2"),
                                    layout: Some(&layout),
                                    module: &module,
                                    entry_point: "main2",
                                    compilation_options: Default::default(),
                                    cache: Default::default(),
                                },
                            )
                    }
                }
            },
            actual
        );
    }
}
