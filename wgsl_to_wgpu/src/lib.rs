//! # wgsl_to_wgpu
//! wgsl_to_wgpu is an experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Features
//! The [create_shader_module] function is intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! ## Limitations
//! This project supports most WGSL types but doesn't enforce certain key properties such as field alignment.
//! It may be necessary to disable running this function for shaders with unsupported types or features.
//! The current implementation assumes all shader stages are part of a single WGSL source file.
//! Vertex attributes using floating point types in WGSL like `vec2<f32>` are assumed to use
//! float inputs instead of normalized attributes like unorm or snorm integers.
//! Insufficient or innaccurate generated code should be replaced by handwritten implementations as needed.

extern crate wgpu_types as wgpu;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::collections::BTreeMap;
use syn::{Ident, Index};
use thiserror::Error;

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

    let bind_group_data = wgsl::get_bind_group_data(&module)?;
    let shader_stages = wgsl::shader_stages(&module);

    // Write all the structs, including uniforms and entry function inputs.
    let structs = structs(&module, options);
    let bind_groups_module = bind_groups_module(&bind_group_data, shader_stages);
    let vertex_module = vertex_module(&module);
    let compute_module = compute_module(&module);

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

        #create_shader_module

        #create_pipeline_layout
    };
    Ok(pretty_print(output))
}

fn pretty_print(tokens: TokenStream) -> String {
    let file = syn::parse_file(&tokens.to_string()).unwrap();
    prettyplease::unparse(&file)
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{}{}", name, index), Span::call_site())
}

fn compute_module(module: &naga::Module) -> TokenStream {
    let workgroup_sizes: Vec<_> = module
        .entry_points
        .iter()
        .filter_map(|e| {
            if e.stage == naga::ShaderStage::Compute {
                // Use Index to avoid specifying the type on literals.
                let name = Ident::new(
                    &format!("{}_WORKGROUP_SIZE", e.name.to_uppercase()),
                    Span::call_site(),
                );
                let [x, y, z] = e.workgroup_size.map(|s| Index::from(s as usize));
                Some(quote!(pub const #name: [u32; 3] = [#x, #y, #z];))
            } else {
                None
            }
        })
        .collect();

    if workgroup_sizes.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod compute {
                #(#workgroup_sizes)*
            }
        }
    }
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
                let location = Index::from(*location as usize);
                let format = wgsl::vertex_format(&module.types[m.ty]);
                // TODO: Will the debug implementation always work with the macro?
                let format = syn::Ident::new(&format!("{:?}", format), Span::call_site());
                quote!(#location => #format)
            })
            .collect();

        // The vertex input structs should already be written at this point.
        // TODO: Support vertex inputs that aren't in a struct.
        quote! {
            impl super::#name {
                pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; #count] = wgpu::vertex_attr_array![#(#attributes),*];
            }
        }
    }).collect()
}

// TODO: Take an iterator instead?
fn bind_groups_module(
    bind_group_data: &BTreeMap<u32, wgsl::GroupData>,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let bind_groups: Vec<_> = bind_group_data
        .iter()
        .map(|(group_no, group)| {
            let group_name = indexed_name_to_ident("BindGroup", *group_no);

            let layout = bind_group_layout(*group_no, group);
            let layout_descriptor = bind_group_layout_descriptor(*group_no, group, shader_stages);
            let group_impl = bind_group(*group_no, group, shader_stages);

            quote! {
                pub struct #group_name(wgpu::BindGroup);
                #layout
                #layout_descriptor
                #group_impl
            }
        })
        .collect();

    let bind_group_fields: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group_name = indexed_name_to_ident("BindGroup", *group_no);
            let field = indexed_name_to_ident("bind_group", *group_no);
            quote!(pub #field: &'a #group_name)
        })
        .collect();

    // TODO: Support compute shader with vertex/fragment in the same module?
    let is_compute = shader_stages == wgpu::ShaderStages::COMPUTE;
    let set_bind_groups = set_bind_groups(bind_group_data, is_compute);

    if bind_groups.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod bind_groups {
                #(#bind_groups)*

                pub struct BindGroups<'a> {
                    #(#bind_group_fields),*
                }

                #set_bind_groups
            }
        }
    }
}

fn set_bind_groups(
    bind_group_data: &BTreeMap<u32, wgsl::GroupData>,
    is_compute: bool,
) -> TokenStream {
    let render_pass = if is_compute {
        quote!(wgpu::ComputePass<'a>)
    } else {
        quote!(wgpu::RenderPass<'a>)
    };

    // The set function for each bind group already sets the index.
    let groups: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("bind_group", *group_no);
            quote!(bind_groups.#group.set(pass);)
        })
        .collect();

    quote! {
        pub fn set_bind_groups<'a>(
            pass: &mut #render_pass,
            bind_groups: BindGroups<'a>,
        ) {
            #(#groups)*
        }
    }
}

fn structs(module: &naga::Module, options: WriteOptions) -> Vec<TokenStream> {
    // Create matching Rust structs for WGSL structs.
    // This is a UniqueArena, so each struct will only be generated once.
    module
        .types
        .iter()
        .filter(|(h, _)| {
            // Shader stage outputs don't need to be instantiated by the user.
            // Many builtin outputs also don't satisfy alignment requirements.
            // Skipping these structs avoids issues deriving encase or bytemuck.
            !module
                .entry_points
                .iter()
                .any(|e| e.function.result.as_ref().map(|r| r.ty) == Some(*h))
        })
        .filter_map(|(_, t)| {
            if let naga::TypeInner::Struct { members, .. } = &t.inner {
                let name = Ident::new(t.name.as_ref().unwrap(), Span::call_site());
                let members = struct_members(members, module, options.matrix_vector_types);
                let mut derives = vec![
                    quote!(Debug),
                    quote!(Copy),
                    quote!(Clone),
                    quote!(PartialEq),
                ];
                if options.derive_bytemuck {
                    derives.push(quote!(bytemuck::Pod));
                    derives.push(quote!(bytemuck::Zeroable));
                }
                if options.derive_encase {
                    derives.push(quote!(encase::ShaderType));
                }
                Some(quote! {
                    #[repr(C)]
                    #[derive(#(#derives),*)]
                    pub struct #name {
                        #(#members),*
                    }
                })
            } else {
                None
            }
        })
        .collect()
}

fn struct_members(
    members: &[naga::StructMember],
    module: &naga::Module,
    format: MatrixVectorTypes,
) -> Vec<TokenStream> {
    members
        .iter()
        .map(|member| {
            let member_name = Ident::new(member.name.as_ref().unwrap(), Span::call_site());
            let member_type = wgsl::rust_type(module, &module.types[member.ty], format);
            quote!(pub #member_name: #member_type)
        })
        .collect()
}

fn bind_group_layout(group_no: u32, group: &wgsl::GroupData) -> TokenStream {
    let fields: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| {
            let field_name = Ident::new(binding.name.as_ref().unwrap(), Span::call_site());
            // TODO: Support more types.
            let field_type = match binding.binding_type.inner {
                naga::TypeInner::Struct { .. } => quote!(wgpu::BufferBinding<'a>),
                naga::TypeInner::Image { .. } => quote!(&'a wgpu::TextureView),
                naga::TypeInner::Sampler { .. } => quote!(&'a wgpu::Sampler),
                naga::TypeInner::Array { .. } => quote!(wgpu::BufferBinding<'a>),
                _ => panic!("Unsupported type for binding fields."),
            };
            quote!(pub #field_name: #field_type)
        })
        .collect();

    let name = indexed_name_to_ident("BindGroupLayout", group_no);
    quote! {
        pub struct #name<'a> {
            #(#fields),*
        }
    }
}

fn bind_group_layout_descriptor(
    group_no: u32,
    group: &wgsl::GroupData,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let entries: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| bind_group_layout_entry(binding, shader_stages))
        .collect();

    let name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", group_no);
    quote! {
        const #name: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                #(#entries),*
            ],
        };
    }
}

fn bind_group_layout_entry(
    binding: &wgsl::GroupBinding,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    // TODO: Assume storage is only used for compute?
    // TODO: Support just vertex or fragment?
    // TODO: Visible from all stages?
    let stages = match shader_stages {
        wgpu::ShaderStages::VERTEX_FRAGMENT => quote!(wgpu::ShaderStages::VERTEX_FRAGMENT),
        wgpu::ShaderStages::COMPUTE => quote!(wgpu::ShaderStages::COMPUTE),
        wgpu::ShaderStages::VERTEX => quote!(wgpu::ShaderStages::VERTEX),
        wgpu::ShaderStages::FRAGMENT => quote!(wgpu::ShaderStages::FRAGMENT),
        _ => todo!(),
    };

    let binding_index = Index::from(binding.binding_index as usize);
    // TODO: Support more types.
    let binding_type = match binding.binding_type.inner {
        naga::TypeInner::Struct { .. } => {
            let buffer_binding_type = wgsl::buffer_binding_type(binding.address_space);

            quote!(wgpu::BindingType::Buffer {
                ty: #buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            })
        }
        naga::TypeInner::Array { .. } => {
            let buffer_binding_type = wgsl::buffer_binding_type(binding.address_space);

            quote!(wgpu::BindingType::Buffer {
                ty: #buffer_binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            })
        }
        naga::TypeInner::Image { dim, class, .. } => {
            let view_dim = match dim {
                naga::ImageDimension::D1 => quote!(wgpu::TextureViewDimension::D1),
                naga::ImageDimension::D2 => quote!(wgpu::TextureViewDimension::D2),
                naga::ImageDimension::D3 => quote!(wgpu::TextureViewDimension::D3),
                naga::ImageDimension::Cube => quote!(wgpu::TextureViewDimension::Cube),
            };

            let sample_type = match class {
                naga::ImageClass::Sampled { kind: _, multi: _ } => {
                    quote!(wgpu::TextureSampleType::Float { filterable: true })
                }
                naga::ImageClass::Depth { multi: _ } => quote!(wgpu::TextureSampleType::Depth),
                naga::ImageClass::Storage {
                    format: _,
                    access: _,
                } => todo!(),
            };

            quote!(wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: #view_dim,
                sample_type: #sample_type,
            })
        }
        naga::TypeInner::Sampler { comparison } => {
            let sampler_type = if comparison {
                quote!(wgpu::SamplerBindingType::Comparison)
            } else {
                quote!(wgpu::SamplerBindingType::Filtering)
            };
            quote!(wgpu::BindingType::Sampler(#sampler_type))
        }
        // TODO: Better error handling.
        _ => panic!("Failed to generate BindingType."),
    };

    quote! {
        wgpu::BindGroupLayoutEntry {
            binding: #binding_index,
            visibility: #stages,
            ty: #binding_type,
            count: None,
        }
    }
}

fn bind_group(
    group_no: u32,
    group: &wgsl::GroupData,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let entries: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| {
            let binding_index = Index::from(binding.binding_index as usize);
            let binding_name = Ident::new(binding.name.as_ref().unwrap(), Span::call_site());
            let resource_type = match binding.binding_type.inner {
                naga::TypeInner::Struct { .. } => {
                    quote!(wgpu::BindingResource::Buffer(bindings.#binding_name))
                }
                naga::TypeInner::Array { .. } => {
                    quote!(wgpu::BindingResource::Buffer(bindings.#binding_name))
                }
                naga::TypeInner::Image { .. } => {
                    quote!(wgpu::BindingResource::TextureView(bindings.#binding_name))
                }
                naga::TypeInner::Sampler { .. } => {
                    quote!(wgpu::BindingResource::Sampler(bindings.#binding_name))
                }
                // TODO: Better error handling.
                _ => panic!("Failed to generate BindingType."),
            };

            quote! {
                wgpu::BindGroupEntry {
                    binding: #binding_index,
                    resource: #resource_type,
                }
            }
        })
        .collect();

    // TODO: Support compute shader with vertex/fragment in the same module?
    let is_compute = shader_stages == wgpu::ShaderStages::COMPUTE;

    let render_pass = if is_compute {
        quote!(wgpu::ComputePass<'a>)
    } else {
        quote!(wgpu::RenderPass<'a>)
    };

    let bind_group_name = indexed_name_to_ident("BindGroup", group_no);
    let bind_group_layout_name = indexed_name_to_ident("BindGroupLayout", group_no);

    let layout_descriptor_name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", group_no);

    let group_no = Index::from(group_no as usize);

    quote! {
        impl #bind_group_name {
            pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&#layout_descriptor_name)
            }

            pub fn from_bindings(device: &wgpu::Device, bindings: #bind_group_layout_name) -> Self {
                let bind_group_layout = device.create_bind_group_layout(&#layout_descriptor_name);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        #(#entries),*
                    ],
                    label: None,
                });
                Self(bind_group)
            }

            pub fn set<'a>(&'a self, render_pass: &mut #render_pass) {
                render_pass.set_bind_group(#group_no, &self.0, &[]);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[test]
    fn write_all_structs_rust() {
        let source = indoc! {r#"
            struct Scalars {
                a: u32,
                b: i32,
                c: f32,
            };

            struct VectorsU8 {
                a: vec2<u8>,
                b: vec3<u8>,
                c: vec4<u8>,
            };

            struct VectorsU16 {
                a: vec2<u16>,
                b: vec3<u16>,
                c: vec4<u16>,
            };

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
            };

            struct VectorsI8 {
                a: vec2<i8>,
                b: vec3<i8>,
                c: vec4<i8>,
            };

            struct VectorsI16 {
                a: vec2<i16>,
                b: vec3<i16>,
                c: vec4<i16>,
            };

            struct VectorsI32 {
                a: vec2<i32>,
                b: vec3<i32>,
                c: vec4<i32>,
            };

            struct VectorsF32 {
                a: vec2<f32>,
                b: vec3<f32>,
                c: vec4<f32>,
            };

            struct VectorsF64 {
                a: vec2<f64>,
                b: vec3<f64>,
                c: vec4<f64>,
            };

            struct MatricesF32 {
                a: mat4x4<f32>,
                b: mat4x3<f32>,
                c: mat4x2<f32>,
                d: mat3x4<f32>,
                e: mat3x3<f32>,
                f: mat3x2<f32>,
                g: mat2x4<f32>,
                h: mat2x3<f32>,
                i: mat2x2<f32>,
            };

            struct MatricesF64 {
                a: mat4x4<f64>,
                b: mat4x3<f64>,
                c: mat4x2<f64>,
                d: mat3x4<f64>,
                e: mat3x3<f64>,
                f: mat3x2<f64>,
                g: mat2x4<f64>,
                h: mat2x3<f64>,
                i: mat2x2<f64>,
            };

            struct StaticArrays {
                a: array<u32, 5>,
                b: array<f32, 3>,
                c: array<mat4x4<f32>, 512>,
            };

            struct Nested {
                a: MatricesF32,
                b: MatricesF64
            }

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(&module, WriteOptions::default());
        let actual = pretty_print(quote!(#(#structs)*));

        assert_eq!(
            indoc! {
                r"
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Scalars {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU8 {
                    pub a: [u8; 2],
                    pub b: [u8; 3],
                    pub c: [u8; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU16 {
                    pub a: [u16; 2],
                    pub b: [u16; 3],
                    pub c: [u16; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU32 {
                    pub a: [u32; 2],
                    pub b: [u32; 3],
                    pub c: [u32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI8 {
                    pub a: [i8; 2],
                    pub b: [i8; 3],
                    pub c: [i8; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI16 {
                    pub a: [i16; 2],
                    pub b: [i16; 3],
                    pub c: [i16; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI32 {
                    pub a: [i32; 2],
                    pub b: [i32; 3],
                    pub c: [i32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsF32 {
                    pub a: [f32; 2],
                    pub b: [f32; 3],
                    pub c: [f32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsF64 {
                    pub a: [f64; 2],
                    pub b: [f64; 3],
                    pub c: [f64; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct MatricesF32 {
                    pub a: [[f32; 4]; 4],
                    pub b: [[f32; 4]; 3],
                    pub c: [[f32; 4]; 2],
                    pub d: [[f32; 3]; 4],
                    pub e: [[f32; 3]; 3],
                    pub f: [[f32; 3]; 2],
                    pub g: [[f32; 2]; 4],
                    pub h: [[f32; 2]; 3],
                    pub i: [[f32; 2]; 2],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct MatricesF64 {
                    pub a: [[f64; 4]; 4],
                    pub b: [[f64; 4]; 3],
                    pub c: [[f64; 4]; 2],
                    pub d: [[f64; 3]; 4],
                    pub e: [[f64; 3]; 3],
                    pub f: [[f64; 3]; 2],
                    pub g: [[f64; 2]; 4],
                    pub h: [[f64; 2]; 3],
                    pub i: [[f64; 2]; 2],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct StaticArrays {
                    pub a: [u32; 5],
                    pub b: [f32; 3],
                    pub c: [[[f32; 4]; 4]; 512],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Nested {
                    pub a: MatricesF32,
                    pub b: MatricesF64,
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_glam() {
        let source = indoc! {r#"
            struct Scalars {
                a: u32,
                b: i32,
                c: f32,
            };

            struct VectorsU8 {
                a: vec2<u8>,
                b: vec3<u8>,
                c: vec4<u8>,
            };

            struct VectorsU16 {
                a: vec2<u16>,
                b: vec3<u16>,
                c: vec4<u16>,
            };

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
            };

            struct VectorsI8 {
                a: vec2<i8>,
                b: vec3<i8>,
                c: vec4<i8>,
            };

            struct VectorsI16 {
                a: vec2<i16>,
                b: vec3<i16>,
                c: vec4<i16>,
            };

            struct VectorsI32 {
                a: vec2<i32>,
                b: vec3<i32>,
                c: vec4<i32>,
            };

            struct VectorsF32 {
                a: vec2<f32>,
                b: vec3<f32>,
                c: vec4<f32>,
            };

            struct VectorsF64 {
                a: vec2<f64>,
                b: vec3<f64>,
                c: vec4<f64>,
            };

            struct MatricesF32 {
                a: mat4x4<f32>,
                b: mat4x3<f32>,
                c: mat4x2<f32>,
                d: mat3x4<f32>,
                e: mat3x3<f32>,
                f: mat3x2<f32>,
                g: mat2x4<f32>,
                h: mat2x3<f32>,
                i: mat2x2<f32>,
            };

            struct MatricesF64 {
                a: mat4x4<f64>,
                b: mat4x3<f64>,
                c: mat4x2<f64>,
                d: mat3x4<f64>,
                e: mat3x3<f64>,
                f: mat3x2<f64>,
                g: mat2x4<f64>,
                h: mat2x3<f64>,
                i: mat2x2<f64>,
            };

            struct StaticArrays {
                a: array<u32, 5>,
                b: array<f32, 3>,
                c: array<mat4x4<f32>, 512>,
            };

            struct Nested {
                a: MatricesF32,
                b: MatricesF64
            }

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                matrix_vector_types: MatrixVectorTypes::Glam,
                ..Default::default()
            },
        );
        let actual = pretty_print(quote!(#(#structs)*));

        assert_eq!(
            indoc! {
                r"
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Scalars {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU8 {
                    pub a: [u8; 2],
                    pub b: [u8; 3],
                    pub c: [u8; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU16 {
                    pub a: [u16; 2],
                    pub b: [u16; 3],
                    pub c: [u16; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU32 {
                    pub a: glam::UVec2,
                    pub b: glam::UVec3,
                    pub c: glam::UVec4,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI8 {
                    pub a: [i8; 2],
                    pub b: [i8; 3],
                    pub c: [i8; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI16 {
                    pub a: [i16; 2],
                    pub b: [i16; 3],
                    pub c: [i16; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI32 {
                    pub a: glam::IVec2,
                    pub b: glam::IVec3,
                    pub c: glam::IVec4,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsF32 {
                    pub a: glam::Vec2,
                    pub b: glam::Vec3,
                    pub c: glam::Vec4,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsF64 {
                    pub a: glam::DVec2,
                    pub b: glam::DVec3,
                    pub c: glam::DVec4,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct MatricesF32 {
                    pub a: glam::Mat4,
                    pub b: [[f32; 4]; 3],
                    pub c: [[f32; 4]; 2],
                    pub d: [[f32; 3]; 4],
                    pub e: glam::Mat3,
                    pub f: [[f32; 3]; 2],
                    pub g: [[f32; 2]; 4],
                    pub h: [[f32; 2]; 3],
                    pub i: glam::Mat2,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct MatricesF64 {
                    pub a: glam::DMat4,
                    pub b: [[f64; 4]; 3],
                    pub c: [[f64; 4]; 2],
                    pub d: [[f64; 3]; 4],
                    pub e: glam::DMat3,
                    pub f: [[f64; 3]; 2],
                    pub g: [[f64; 2]; 4],
                    pub h: [[f64; 2]; 3],
                    pub i: glam::DMat2,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct StaticArrays {
                    pub a: [u32; 5],
                    pub b: [f32; 3],
                    pub c: [glam::Mat4; 512],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Nested {
                    pub a: MatricesF32,
                    pub b: MatricesF64,
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_encase_bytemuck() {
        let source = indoc! {r#"
            struct Input0 {
                a: u32,
                b: i32,
                c: f32,
            };

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: true,
                derive_bytemuck: true,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = pretty_print(quote!(#(#structs)*));

        assert_eq!(
            indoc! {
                r"
                #[repr(C)]
                #[derive(
                    Debug,
                    Copy,
                    Clone,
                    PartialEq,
                    bytemuck::Pod,
                    bytemuck::Zeroable,
                    encase::ShaderType
                )]
                pub struct Input0 {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_skip_stage_outputs() {
        let source = indoc! {r#"
            struct Input0 {
                a: u32,
                b: i32,
                c: f32,
            };

            struct Output0 {
                a: f32
            }

            @fragment
            fn main() -> Output0 {
                var out: Output0;
                return out;
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: false,
                derive_bytemuck: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = pretty_print(quote!(#(#structs)*));

        assert_eq!(
            indoc! {
                r"
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Input0 {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                "
            },
            actual
        );
    }

    #[test]
    fn bind_groups_module_compute() {
        let source = indoc! {r#"
            struct VertexInput0 {};
            struct VertexWeight {};
            struct Vertices {};
            struct VertexWeights {};
            struct Transforms {};

            @group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read> vertex_weights: VertexWeights;
            @group(0) @binding(2) var<storage, read_write> dst: Vertices;

            @group(1) @binding(0) var<uniform> transforms: Transforms;

            @compute
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module).unwrap();

        let actual = pretty_print(bind_groups_module(
            &bind_group_data,
            wgpu::ShaderStages::COMPUTE,
        ));

        assert_eq!(
            indoc! {
                r#"
                pub mod bind_groups {
                    pub struct BindGroup0(wgpu::BindGroup);
                    pub struct BindGroupLayout0<'a> {
                        pub src: wgpu::BufferBinding<'a>,
                        pub vertex_weights: wgpu::BufferBinding<'a>,
                        pub dst: wgpu::BufferBinding<'a>,
                    }
                    const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage {
                                        read_only: true,
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage {
                                        read_only: true,
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage {
                                        read_only: false,
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    };
                    impl BindGroup0 {
                        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
                        }
                        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
                            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
                            let bind_group = device
                                .create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(bindings.src),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::Buffer(
                                                    bindings.vertex_weights,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: wgpu::BindingResource::Buffer(bindings.dst),
                                            },
                                        ],
                                        label: None,
                                    },
                                );
                            Self(bind_group)
                        }
                        pub fn set<'a>(&'a self, render_pass: &mut wgpu::ComputePass<'a>) {
                            render_pass.set_bind_group(0, &self.0, &[]);
                        }
                    }
                    pub struct BindGroup1(wgpu::BindGroup);
                    pub struct BindGroupLayout1<'a> {
                        pub transforms: wgpu::BufferBinding<'a>,
                    }
                    const LAYOUT_DESCRIPTOR1: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    };
                    impl BindGroup1 {
                        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR1)
                        }
                        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout1) -> Self {
                            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR1);
                            let bind_group = device
                                .create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(bindings.transforms),
                                            },
                                        ],
                                        label: None,
                                    },
                                );
                            Self(bind_group)
                        }
                        pub fn set<'a>(&'a self, render_pass: &mut wgpu::ComputePass<'a>) {
                            render_pass.set_bind_group(1, &self.0, &[]);
                        }
                    }
                    pub struct BindGroups<'a> {
                        pub bind_group0: &'a BindGroup0,
                        pub bind_group1: &'a BindGroup1,
                    }
                    pub fn set_bind_groups<'a>(
                        pass: &mut wgpu::ComputePass<'a>,
                        bind_groups: BindGroups<'a>,
                    ) {
                        bind_groups.bind_group0.set(pass);
                        bind_groups.bind_group1.set(pass);
                    }
                }
                "#
            },
            actual
        );
    }

    #[test]
    fn bind_groups_module_vertex_fragment() {
        // Test different texture and sampler types.
        let source = indoc! {r#"
            struct Transforms {};

            @group(0) @binding(0)
            var color_texture: texture_2d<f32>;
            @group(0) @binding(1)
            var color_sampler: sampler;
            @group(0) @binding(2)
            var depth_texture: texture_depth_2d;
            @group(0) @binding(3)
            var comparison_sampler: sampler_comparison;

            @group(1) @binding(0) var<uniform> transforms: Transforms;

            @vertex
            fn vs_main() {}

            @fragment
            fn fs_main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module).unwrap();

        let actual = pretty_print(bind_groups_module(
            &bind_group_data,
            wgpu::ShaderStages::VERTEX_FRAGMENT,
        ));

        // TODO: Are storage buffers valid for vertex/fragment?
        assert_eq!(
            indoc! {
                r#"
                pub mod bind_groups {
                    pub struct BindGroup0(wgpu::BindGroup);
                    pub struct BindGroupLayout0<'a> {
                        pub color_texture: &'a wgpu::TextureView,
                        pub color_sampler: &'a wgpu::Sampler,
                        pub depth_texture: &'a wgpu::TextureView,
                        pub comparison_sampler: &'a wgpu::Sampler,
                    }
                    const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    multisampled: false,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    multisampled: false,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    sample_type: wgpu::TextureSampleType::Depth,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                                count: None,
                            },
                        ],
                    };
                    impl BindGroup0 {
                        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
                        }
                        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
                            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
                            let bind_group = device
                                .create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::TextureView(
                                                    bindings.color_texture,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::Sampler(
                                                    bindings.color_sampler,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: wgpu::BindingResource::TextureView(
                                                    bindings.depth_texture,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 3,
                                                resource: wgpu::BindingResource::Sampler(
                                                    bindings.comparison_sampler,
                                                ),
                                            },
                                        ],
                                        label: None,
                                    },
                                );
                            Self(bind_group)
                        }
                        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
                            render_pass.set_bind_group(0, &self.0, &[]);
                        }
                    }
                    pub struct BindGroup1(wgpu::BindGroup);
                    pub struct BindGroupLayout1<'a> {
                        pub transforms: wgpu::BufferBinding<'a>,
                    }
                    const LAYOUT_DESCRIPTOR1: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    };
                    impl BindGroup1 {
                        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR1)
                        }
                        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout1) -> Self {
                            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR1);
                            let bind_group = device
                                .create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(bindings.transforms),
                                            },
                                        ],
                                        label: None,
                                    },
                                );
                            Self(bind_group)
                        }
                        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
                            render_pass.set_bind_group(1, &self.0, &[]);
                        }
                    }
                    pub struct BindGroups<'a> {
                        pub bind_group0: &'a BindGroup0,
                        pub bind_group1: &'a BindGroup1,
                    }
                    pub fn set_bind_groups<'a>(
                        pass: &mut wgpu::RenderPass<'a>,
                        bind_groups: BindGroups<'a>,
                    ) {
                        bind_groups.bind_group0.set(pass);
                        bind_groups.bind_group1.set(pass);
                    }
                }
                "#
            },
            actual
        );
    }

    #[test]
    fn bind_groups_module_vertex() {
        // The actual content of the structs doesn't matter.
        // We only care about the groups and bindings.
        let source = indoc! {r#"
            struct Transforms {};

            @group(0) @binding(0) var<uniform> transforms: Transforms;

            @vertex
            fn vs_main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module).unwrap();

        let actual = pretty_print(bind_groups_module(
            &bind_group_data,
            wgpu::ShaderStages::VERTEX,
        ));

        assert_eq!(
            indoc! {
                r#"
                pub mod bind_groups {
                    pub struct BindGroup0(wgpu::BindGroup);
                    pub struct BindGroupLayout0<'a> {
                        pub transforms: wgpu::BufferBinding<'a>,
                    }
                    const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    };
                    impl BindGroup0 {
                        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
                        }
                        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
                            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
                            let bind_group = device
                                .create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(bindings.transforms),
                                            },
                                        ],
                                        label: None,
                                    },
                                );
                            Self(bind_group)
                        }
                        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
                            render_pass.set_bind_group(0, &self.0, &[]);
                        }
                    }
                    pub struct BindGroups<'a> {
                        pub bind_group0: &'a BindGroup0,
                    }
                    pub fn set_bind_groups<'a>(
                        pass: &mut wgpu::RenderPass<'a>,
                        bind_groups: BindGroups<'a>,
                    ) {
                        bind_groups.bind_group0.set(pass);
                    }
                }
                "#
            },
            actual
        );
    }

    #[test]
    fn bind_groups_module_fragment() {
        // The actual content of the structs doesn't matter.
        // We only care about the groups and bindings.
        let source = indoc! {r#"
            struct Transforms {};

            @group(0) @binding(0) var<uniform> transforms: Transforms;

            @fragment
            fn fs_main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module).unwrap();

        let actual = pretty_print(bind_groups_module(
            &bind_group_data,
            wgpu::ShaderStages::FRAGMENT,
        ));

        assert_eq!(
            indoc! {
                r#"
                pub mod bind_groups {
                    pub struct BindGroup0(wgpu::BindGroup);
                    pub struct BindGroupLayout0<'a> {
                        pub transforms: wgpu::BufferBinding<'a>,
                    }
                    const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    };
                    impl BindGroup0 {
                        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
                        }
                        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
                            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
                            let bind_group = device
                                .create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(bindings.transforms),
                                            },
                                        ],
                                        label: None,
                                    },
                                );
                            Self(bind_group)
                        }
                        pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
                            render_pass.set_bind_group(0, &self.0, &[]);
                        }
                    }
                    pub struct BindGroups<'a> {
                        pub bind_group0: &'a BindGroup0,
                    }
                    pub fn set_bind_groups<'a>(
                        pass: &mut wgpu::RenderPass<'a>,
                        bind_groups: BindGroups<'a>,
                    ) {
                        bind_groups.bind_group0.set(pass);
                    }
                }
                "#
            },
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
    fn set_bind_groups_vertex_fragment() {
        let source = indoc! {r#"
            struct Transforms {};

            @group(0) @binding(0) var color_texture: texture_2d<f32>;
            @group(1) @binding(0) var<uniform> transforms: Transforms;

            @vertex
            fn vs_main() {}

            @fragment
            fn fs_main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module).unwrap();

        let actual = pretty_print(set_bind_groups(&bind_group_data, false));

        assert_eq!(
            indoc! {
                r"
            pub fn set_bind_groups<'a>(
                pass: &mut wgpu::RenderPass<'a>,
                bind_groups: BindGroups<'a>,
            ) {
                bind_groups.bind_group0.set(pass);
                bind_groups.bind_group1.set(pass);
            }
            "
            },
            actual
        );
    }

    #[test]
    fn set_bind_groups_compute() {
        let source = indoc! {r#"
            struct Transforms {};

            @group(0) @binding(0)
            var color_texture: texture_2d<f32>;
            @group(1) @binding(0) var<uniform> transforms: Transforms;

            @compute
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module).unwrap();

        let actual = pretty_print(set_bind_groups(&bind_group_data, true));

        // The only change is that the function takes a ComputePass instead.
        assert_eq!(
            indoc! {
                r"
            pub fn set_bind_groups<'a>(
                pass: &mut wgpu::ComputePass<'a>,
                bind_groups: BindGroups<'a>,
            ) {
                bind_groups.bind_group0.set(pass);
                bind_groups.bind_group1.set(pass);
            }
            "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_empty() {
        let source = indoc! {r#"
            @vertex
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pretty_print(vertex_module(&module));

        assert_eq!("", actual);
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
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
                            0 => Float32, 1 => Float32x2, 2 => Float32x3, 3 => Float32x4
                        ];
                    }
                }
                "
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
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
                            0 => Float64, 1 => Float64x2, 2 => Float64x3, 3 => Float64x4
                        ];
                    }
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_sint8() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: vec2<i8>,
                @location(1) a: vec4<i8>,

            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
                            0 => Sint8x2, 1 => Sint8x4
                        ];
                    }
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_sint16() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: vec2<i16>,
                @location(1) a: vec4<i16>,

            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
                            0 => Sint16x2, 1 => Sint16x4
                        ];
                    }
                }
                "
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
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
                            0 => Sint32, 1 => Sint32x2, 2 => Sint32x3, 3 => Sint32x4
                        ];
                    }
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_uint8() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: vec2<u8>,
                @location(1) a: vec4<u8>,

            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
                            0 => Uint8x2, 1 => Uint8x4
                        ];
                    }
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_uint16() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: vec2<u16>,
                @location(1) a: vec4<u16>,

            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
                            0 => Uint16x2, 1 => Uint16x4
                        ];
                    }
                }
                "
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
        let actual = pretty_print(vertex_module(&module));

        assert_eq!(
            indoc! {
                r"
                pub mod vertex {
                    impl super::VertexInput0 {
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
                            0 => Uint32, 1 => Uint32x2, 2 => Uint32x3, 3 => Uint32x4
                        ];
                    }
                }
                "
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
        let actual = pretty_print(compute_module(&module));

        assert_eq!("", actual);
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
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = pretty_print(compute_module(&module));

        assert_eq!(
            indoc! {
            r"
            pub mod compute {
                pub const MAIN1_WORKGROUP_SIZE: [u32; 3] = [1, 2, 3];
                pub const MAIN2_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
            }
            "
            },
            actual
        );
    }
}
