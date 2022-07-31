//! # wgsl_to_wgpu
//! wgsl_to_wgpu is an experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Features
//! The [create_shader_module] function is intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! ## Limitations
//! This project currently supports a small subset of WGSL types and doesn't enforce certain key properties such as field alignment.
//! It may be necessary to disable running this function for shaders with unsupported types or features.
//! The current implementation assumes all shader stages are part of a single WGSL source file.

extern crate wgpu_types as wgpu;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::collections::BTreeMap;
use syn::{Ident, Index};
use thiserror::Error;

mod wgsl;

// TODO: Simplify these templates and indentation?
// TODO: Structure the code to make it easier to imagine what the output will look like.
/// Errors while generating Rust source for a WGSl shader module.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum CreateModuleError {
    /// Bind group sets must be consecutive and start from 0.
    /// See `bind_group_layouts` for [wgpu::PipelineLayoutDescriptor].
    #[error("bind groups are non-consecutive or do not start from 0")]
    NonConsecutiveBindGroups,

    /// Each binding resource must be associated with exactly one binding index.
    #[error("duplicate binding found with index `{binding}`")]
    DuplicateBinding { binding: u32 },
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
    let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();
    let text = wgsl_to_wgpu::create_shader_module(&wgsl_source, "shader.wgsl").unwrap();
    std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
}
```
 */
pub fn create_shader_module(
    wgsl_source: &str,
    wgsl_include_path: &str,
) -> Result<String, CreateModuleError> {
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap();

    let bind_group_data = wgsl::get_bind_group_data(&module)?;
    let shader_stages = wgsl::shader_stages(&module);

    // Write all the structs, including uniforms and entry function inputs.
    let structs = structs(&module);
    let bind_groups_module = bind_groups_module(&bind_group_data, shader_stages);
    let vertex_module = vertex_module(&module);

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
        .iter()
        .map(|(group_no, _)| {
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

fn vertex_module(module: &naga::Module) -> TokenStream {
    let structs = vertex_input_structs(module);
    quote! {
        pub mod vertex {
            #(#structs)*
        }
    }
}

fn vertex_input_structs(module: &naga::Module) -> Vec<TokenStream> {
    let vertex_inputs = wgsl::get_vertex_input_structs(module);
    vertex_inputs.iter().map(|input|  {
        let name = Ident::new(&input.name, Span::call_site());

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

        // TODO: Account for alignment/padding?
        let size_in_bytes: u64 = input
            .fields
            .iter()
            .map(|(_, m)| wgsl::vertex_format(&module.types[m.ty]).size())
            .sum();
        let size_in_bytes = Index::from(size_in_bytes as usize);

        // The vertex input structs should already be written at this point.
        // TODO: Support vertex inputs that aren't in a struct.
        quote! {
            impl super::#name {
                pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; #count] = wgpu::vertex_attr_array![#(#attributes),*];
                /// The total size in bytes of all fields without considering padding or alignment.
                pub const SIZE_IN_BYTES: u64 = #size_in_bytes;
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

fn structs(module: &naga::Module) -> Vec<TokenStream> {
    // Create matching Rust structs for WGSL structs.
    // The goal is to eventually have safe ways to initialize uniform buffers.

    // TODO: How to provide a convenient way to work with these types.
    // Users will want to either a) create a new buffer each type or b) reuse an existing buffer.
    // It might not make sense from a performance perspective to constantly create new resources.
    // This requires the user to keep track of the buffer separately from the BindGroup itself.

    // This is a UniqueArena, so types will only be defined once.
    module
        .types
        .iter()
        .filter_map(|(_, t)| {
            if let naga::TypeInner::Struct { members, .. } = &t.inner {
                let name = Ident::new(t.name.as_ref().unwrap(), Span::call_site());
                // TODO: Enforce std140 with crevice for uniform buffers to be safe?
                let members = struct_members(members, module);
                Some(quote! {
                    #[repr(C)]
                    #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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

fn struct_members(members: &[naga::StructMember], module: &naga::Module) -> Vec<TokenStream> {
    members
        .iter()
        .map(|member| {
            let member_name = Ident::new(member.name.as_ref().unwrap(), Span::call_site());
            let member_type = wgsl::rust_type(module, &module.types[member.ty]);
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
                // TODO: Is it possible to make structs strongly typed and handle buffer creation automatically?
                // This could be its own module and associated tests.
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
    fn write_all_structs() {
        let source = indoc! {r#"
            struct VectorsF32 {
                a: vec2<f32>,
                b: vec3<f32>,
                c: vec4<f32>,
            };

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
            };

            struct MatricesF32 {
                a: mat4x4<f32>
            };
            
            struct StaticArrays {
                a: array<u32, 5>,
                b: array<f32, 3>,
                c: array<mat4x4<f32>, 512>,
            };

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(&module);
        let file = syn::parse_file(&quote!(#(#structs)*).to_string()).unwrap();
        let actual = prettyplease::unparse(&file);

        assert_eq!(
            indoc! {
                r"
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct VectorsF32 {
                    pub a: [f32; 2],
                    pub b: [f32; 3],
                    pub c: [f32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct VectorsU32 {
                    pub a: [u32; 2],
                    pub b: [u32; 3],
                    pub c: [u32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct MatricesF32 {
                    pub a: glam::Mat4,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct StaticArrays {
                    pub a: [u32; 5],
                    pub b: [f32; 3],
                    pub c: [glam::Mat4; 512],
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

        create_shader_module(source, "shader.wgsl").unwrap();
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

        let result = create_shader_module(source, "shader.wgsl");
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

        let result = create_shader_module(source, "shader.wgsl");
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

        assert_eq!("pub mod vertex {}\n", actual);
    }

    #[test]
    fn write_vertex_module_single_input_float() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: vec2<f32>,
                @location(1) b: vec3<f32>,
                @location(2) c: vec4<f32>,
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
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
                            0 => Float32x2, 1 => Float32x3, 2 => Float32x4
                        ];
                        /// The total size in bytes of all fields without considering padding or alignment.
                        pub const SIZE_IN_BYTES: u64 = 36;
                    }
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_sint() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: i32,
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
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![
                            0 => Sint32
                        ];
                        /// The total size in bytes of all fields without considering padding or alignment.
                        pub const SIZE_IN_BYTES: u64 = 4;
                    }
                }
                "
            },
            actual
        );
    }

    #[test]
    fn write_vertex_module_single_input_uint() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: vec2<u32>,
                @location(1) b: vec3<u32>,
                @location(2) c: vec4<u32>,
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
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
                            0 => Uint32x2, 1 => Uint32x3, 2 => Uint32x4
                        ];
                        /// The total size in bytes of all fields without considering padding or alignment.
                        pub const SIZE_IN_BYTES: u64 = 36;
                    }
                }
                "
            },
            actual
        );
    }
}
