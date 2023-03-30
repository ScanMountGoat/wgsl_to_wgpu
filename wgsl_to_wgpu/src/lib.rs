//! # wgsl_to_wgpu
//! wgsl_to_wgpu is an experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Features
//! The [create_shader_module] function is intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! Writing Rust code to interact with WGSL shaders can be tedious and error prone,
//! especially when the types and functions in the shader code change during development.
//! wgsl_to_wgpu is not a rendering library and does not offer high level abstractions like a scene graph or material system.
//! Using generated code still has a number of advantages compared to writing the code by hand.
//! The code generated by wgsl_to_wgpu can help with valid API usage like
//! - setting all bind groups and bind group bindings
//! - setting correct struct fields and field types for vertex input buffers
//! - setting correct struct struct fields and field types for storage and uniform buffers
//! - configuring shader initialization
//! - getting vertex attribute offsets for vertex buffers
//! - const validation of struct memory layouts when using bytemuck
//!
//! ## Limitations
//! - It may be necessary to disable running this function for shaders with unsupported types or features.
//! Insufficient or innaccurate generated code should be replaced by handwritten implementations as needed.
//!
//! - The generated code will not prevent accidentally calling a function from an unrelated generated module.
//! It's recommended to name the shader module with the same name as the shader
//! and use unique shader names to avoid issues.
//! Using generated code from a different shader module may be desirable in some cases
//! such as using the same camera struct definition in multiple WGSL shaders.
//!
//! - The current implementation assumes all shader stages are part of a single WGSL source file.
//! Shader modules split across files will be supported in a future release.
//!
//! - Uniform and storage buffers can be initialized using the wrong generated Rust struct.
//! WGPU will still validate the size of the buffer binding at runtime.
//!
//! - Most but not all WGSL types are currently supported.
//!
//! - Vertex attributes using floating point types in WGSL like `vec2<f32>` are assumed to use
//! float inputs instead of normalized attributes like unorm or snorm integers.
//!
//! - It's possible to achieve slightly better performance than the generated code in some cases like
//! avoiding redundant bind group bindings or adjusting resource shader stage visibility.
//! This can be fixed by calling lower level generated functions or handwriting functions as needed.

extern crate wgpu_types as wgpu;

use bindgroup::{bind_groups_module, get_bind_group_data};
use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use naga::ShaderStage;
use syn::{Ident, Index};
use thiserror::Error;

mod bindgroup;
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
    /// Derive [encase::ShaderType](https://docs.rs/encase/latest/encase/trait.ShaderType.html#)
    /// for user defined WGSL structs when `true`.
    pub derive_encase: bool,

    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for user defined WGSL structs when `true`.
    pub derive_bytemuck: bool,

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

/// Parses the WGSL shader from `wgsl_source` and returns the generated Rust module's source code.
///
/// The `wgsl_include_path` should be a valid path for the `include_wgsl!` macro used in the generated file.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/**
```rust no_run
// build.rs
fn main() {
    // Read the shader source file.
    let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();
    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions::default();
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
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap();

    let bind_group_data = get_bind_group_data(&module)?;
    let shader_stages = wgsl::shader_stages(&module);

    // Write all the structs, including uniforms and entry function inputs.
    let structs = structs::structs(&module, options);
    let bind_groups_module = bind_groups_module(&bind_group_data, shader_stages);
    let vertex_module = vertex_module(&module);
    let compute_module = compute_module(&module);
    let entry_point_constants = entry_point_constants(&module);
    let vertex_states = vertex_states(&module);

    let create_shader_module = quote! {
        pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
            let source = std::borrow::Cow::Borrowed(include_str!(#wgsl_include_path));
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

    let create_pipeline_layout = quote! {
        pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    #(&#bind_group_layouts),*
                ],
                push_constant_ranges: &[],
            })
        }
    };

    let output = quote! {
        #(#structs)*

        #bind_groups_module

        #vertex_module

        #compute_module

        #entry_point_constants

        #vertex_states

        #create_shader_module

        #create_pipeline_layout
    };
    Ok(pretty_print(&output))
}

fn pretty_print(tokens: &TokenStream) -> String {
    let file = syn::parse_file(&tokens.to_string()).unwrap();
    prettyplease::unparse(&file)
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{}{}", name, index), Span::call_site())
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

fn vertex_module(module: &naga::Module) -> TokenStream {
    let structs = vertex_input_structs(module);
    if structs.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod vertex {
                #(#structs)*
            }
        }
    }
}

fn entry_point_constants(module: &naga::Module) -> TokenStream {
    let entry_points: Vec<TokenStream> = module.entry_points.iter().map(|entry_point| {
        let entry_name = Literal::string(&entry_point.name);
        let const_name = Ident::new(
            &format!("ENTRY_{}", &entry_point.name.to_uppercase()),
            Span::call_site());
        quote! {
            pub const #const_name: &'static str = #entry_name;
        }
    }).collect();

    quote! {
        #(#entry_points)*
    }
}

fn vertex_states(module: &naga::Module) -> TokenStream {
    let vertex_inputs = wgsl::get_vertex_input_structs(module);
    let mut step_mode_params = vec![];
    let layout_expressions: Vec<TokenStream> = vertex_inputs.iter().enumerate().map(|(idx, s)| {
        let name = Ident::new(&s.name, Span::call_site());
        let step_mode = indexed_name_to_ident("step_mode_", idx as u32);
        let step_mode_clone = step_mode.clone();
        step_mode_params.push(quote!(#step_mode_clone: wgpu::VertexStepMode));
        quote!(#name::vertex_buffer_layout(#step_mode))
    }).collect();

    let vertex_states: Vec<TokenStream> = module.entry_points.iter().map(|entry_point| {
        match &entry_point.stage {
            ShaderStage::Vertex => {
                let fn_name = Ident::new(
                    &format!("{}_vertex_state", &entry_point.name),
                    Span::call_site());
                let const_name = Ident::new(
                    &format!("ENTRY_{}", &entry_point.name.to_uppercase()),
                    Span::call_site());
                quote! {
                    pub fn #fn_name(shader_module: &wgpu::ShaderModule, #(#step_mode_params),*) -> wgpu::VertexState {
                        let vertex_layouts = [
                            #(#layout_expressions),*
                        ];
                        wgpu::VertexState {
                            module: &shader_module,
                            entry_point: #const_name,
                            buffers: &vertex_layouts
                        }
                    }
                }
            }
            _ => quote!()
        }
    }).collect();

    quote! {
        #(#vertex_states)*
    }
}

fn vertex_input_structs(module: &naga::Module) -> Vec<TokenStream> {
    let vertex_inputs = wgsl::get_vertex_input_structs(module);
    vertex_inputs.iter().map(|input|  {
        let name = Ident::new(&input.name, Span::call_site());

        // Use index to avoid adding prefix to literals.
        let count = Index::from(input.fields.len());
        let attributes: Vec<_> = input
            .fields
            .iter()
            .map(|(location, m)| {
                let field_name: TokenStream = m.name.as_ref().unwrap().parse().unwrap();
                let location = Index::from(*location as usize);
                let format = wgsl::vertex_format(&module.types[m.ty]);
                // TODO: Will the debug implementation always work with the macro?
                let format = Ident::new(&format!("{:?}", format), Span::call_site());

                quote! {
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::#format,
                        offset: memoffset::offset_of!(super::#name, #field_name) as u64,
                        shader_location: #location,
                    }
                }
            })
            .collect();


        // The vertex_attr_array! macro doesn't account for field alignment.
        // Structs with glam::Vec4 and glam::Vec3 fields will not be tightly packed.
        // Manually calculate the Rust field offsets to support using bytemuck for vertices.
        // This works since we explicitly mark all generated structs as repr(C).
        // Assume elements are in Rust arrays or slices, so use size_of for stride.
        // TODO: Should this enforce WebGPU alignment requirements for compatibility?
        // https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-gpuvertexbufferlayout

        // TODO: Support vertex inputs that aren't in a struct.
        // TODO: Just add this to the struct directly?
        quote! {
            impl super::#name {
                pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; #count] = [#(#attributes),*];

                pub const fn vertex_buffer_layout(step_mode: wgpu::VertexStepMode) -> wgpu::VertexBufferLayout<'static> {
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<super::#name>() as u64,
                        step_mode,
                        attributes: &super::#name::VERTEX_ATTRIBUTES
                    }
                }
            }
        }
    }).collect()
}

// Tokenstreams can't be compared directly using PartialEq.
// Use pretty_print to normalize the formatting and compare strings.
// Use a colored diff output to make differences easier to see.
#[cfg(test)]
#[macro_export]
macro_rules! assert_tokens_eq {
    ($a:expr, $b:expr) => {
        pretty_assertions::assert_eq!(crate::pretty_print(&$a), crate::pretty_print(&$b));
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use indoc::indoc;

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
        let actual = vertex_module(&module);

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
        let actual = vertex_module(&module);

        assert_tokens_eq!(
            quote! {
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: memoffset::offset_of!(super::VertexInput0, b) as u64,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: memoffset::offset_of!(super::VertexInput0, c) as u64,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: memoffset::offset_of!(super::VertexInput0, d) as u64,
                                shader_location: 3,
                            },
                        ];
                        pub const fn vertex_buffer_layout(
                            step_mode: wgpu::VertexStepMode,
                        ) -> wgpu::VertexBufferLayout<'static> {
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<super::VertexInput0>() as u64,
                                step_mode,
                                attributes: &super::VertexInput0::VERTEX_ATTRIBUTES,
                            }
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
        let actual = vertex_module(&module);

        assert_tokens_eq!(
            quote! {
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float64,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float64x2,
                                offset: memoffset::offset_of!(super::VertexInput0, b) as u64,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float64x3,
                                offset: memoffset::offset_of!(super::VertexInput0, c) as u64,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float64x4,
                                offset: memoffset::offset_of!(super::VertexInput0, d) as u64,
                                shader_location: 3,
                            },
                        ];
                        pub fn vertex_buffer_layout(
                            step_mode: wgpu::VertexStepMode,
                        ) -> wgpu::VertexBufferLayout<'static> {
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<super::VertexInput0>() as u64,
                                step_mode,
                                attributes: &super::VertexInput0::VERTEX_ATTRIBUTES,
                            }
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
        let actual = vertex_module(&module);

        assert_tokens_eq!(
            quote! {
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Sint32,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Sint32x2,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Sint32x3,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Sint32x4,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 3,
                            },
                        ];
                        pub fn vertex_buffer_layout(
                            step_mode: wgpu::VertexStepMode,
                        ) -> wgpu::VertexBufferLayout<'static> {
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<super::VertexInput0>() as u64,
                                step_mode,
                                attributes: &super::VertexInput0::VERTEX_ATTRIBUTES,
                            }
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
        let actual = vertex_module(&module);

        assert_tokens_eq!(
            quote! {
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: memoffset::offset_of!(super::VertexInput0, a) as u64,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x2,
                                offset: memoffset::offset_of!(super::VertexInput0, b) as u64,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x3,
                                offset: memoffset::offset_of!(super::VertexInput0, c) as u64,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x4,
                                offset: memoffset::offset_of!(super::VertexInput0, d) as u64,
                                shader_location: 3,
                            },
                        ];
                        pub fn vertex_buffer_layout(
                            step_mode: wgpu::VertexStepMode,
                        ) -> wgpu::VertexBufferLayout<'static> {
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<super::VertexInput0>() as u64,
                                step_mode,
                                attributes: &super::VertexInput0::VERTEX_ATTRIBUTES,
                            }
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
                                },
                            )
                    }
                }
            },
            actual
        );
    }
}
