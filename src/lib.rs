use std::collections::{BTreeMap};

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{parse_macro_input, LitStr};

#[proc_macro]
pub fn wgsl_module(input: TokenStream) -> TokenStream {
    // TODO: How to handle the errors from naga?
    let input_path = parse_macro_input!(input as LitStr).value();
    // TODO: This won't always use the correct path.
    // TODO: Use include_str! to recompile when the source changes?
    let wgsl_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(input_path);
    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();

    let module = naga::front::wgsl::parse_str(&wgsl_source).unwrap();

    let bind_groups = get_binding_groups(&module);
    let generate_bind_groups: Vec<_> = bind_groups.iter().map(|(group_no, group)| {
        let group_name = Ident::new(&format!("BindGroup{}", group_no), Span::call_site());

        // TODO: Clean this up.
        let binding_fields: Vec<_> = group.bindings.iter().map(generate_binding_field).collect();

        let descriptor_entries = group.bindings.iter().map(
            generate_bind_group_layout_entry
        );

        let bind_group_entries: Vec<_> = group.bindings.iter().map(generate_bind_group_entry).collect();
        
        quote! {
            pub struct #group_name<'a> {
                #(
                    #binding_fields,
                )*
            }

            impl<'a> #group_name<'a> {
                fn get_layout_descriptor() -> wgpu::BindGroupLayoutDescriptor<'a> {
                    wgpu::BindGroupLayoutDescriptor::<'a> {
                        // TODO: Use the name here?
                        label: None,
                        entries: &[
                            #(
                                #descriptor_entries
                            ),*
                        ],
                    }
                }

                pub fn get_bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
                    device.create_bind_group_layout(&#group_name::get_layout_descriptor())
                }

                pub fn get_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
                    // TODO: Avoid creating this more than once?
                    let bind_group_layout = device.create_bind_group_layout(&Self::get_layout_descriptor());
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            #(
                                #bind_group_entries
                            ),*
                        ],
                        label: None,
                    })
                }
            }
        }
    }).collect();

    let expanded = quote! {
        mod shader_types {
            pub mod bind_groups {
                #(
                    #generate_bind_groups
                )*
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_bind_group_entry(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let binding_index = group_binding.binding_index;
    let binding_name = Ident::new(&format!("binding{}", binding_index), Span::call_site());
    let resource_type = match group_binding.inner_type {
        naga::TypeInner::Struct { .. } => quote! {
            // TODO: Don't assume the entire buffer is used.
            self.#binding_name.as_entire_binding()
        },
        naga::TypeInner::Image { .. } => quote! {
            wgpu::BindingResource::TextureView(self.#binding_name)
        },
        naga::TypeInner::Sampler { .. } => quote! {
            wgpu::BindingResource::Sampler(self.#binding_name)
        },
        // TODO: Better error handling.
        _ => panic!("Failed to generate BindingType."),
    };

    quote! {
        wgpu::BindGroupEntry {
            binding: #binding_index,
            // TODO: Set the type
            resource: #resource_type,
        }
    }
}

fn generate_bind_group_layout_entry(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let binding_index = group_binding.binding_index;
    let binding_type = match group_binding.inner_type {
        naga::TypeInner::Struct { .. } => quote! {wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        },
        naga::TypeInner::Image { .. } => quote! {
            // TODO: Don't assume the dimensions.
            wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            }
        },
        naga::TypeInner::Sampler { .. } => quote! {
            wgpu::BindingType::Sampler {
                comparison: false,
                filtering: true,
            }
        },
        // TODO: Better error handling.
        _ => panic!("Failed to generate BindingType."),
    };

    quote! {
        wgpu::BindGroupLayoutEntry {
            binding: #binding_index,
            // TODO: This can't  be determined easily from WGSL, so just use both (missing compute).
            visibility: wgpu::ShaderStage::VERTEX_FRAGMENT,
            ty: #binding_type,
            // TODO: Support arrays?
            count: None,
        }
    }
}

fn generate_binding_field(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let field_name = Ident::new(
        &format!("binding{}", group_binding.binding_index),
        Span::call_site(),
    );
    // TODO: Support more types.
    let field_type = match group_binding.inner_type {
        naga::TypeInner::Struct { .. } => quote! { &'a wgpu::Buffer },
        naga::TypeInner::Image { .. } => quote! { &'a wgpu::TextureView },
        naga::TypeInner::Sampler { .. } => quote! { &'a wgpu::Sampler },
        _ => panic!("Unsupported type for binding fields."),
    };

    quote! {
        pub #field_name: #field_type
    }

}

struct GroupData<'a> {
    bindings: Vec<GroupBinding<'a>>,
}

struct GroupBinding<'a> {
    binding_index: u32,
    inner_type: &'a naga::TypeInner,
}

fn get_binding_groups(module: &naga::Module) -> BTreeMap<u32, GroupData> {
    // Use a BTree to sort type and field names by group index.
    // This isn't strictly necessary but makes the generated code cleaner.
    let mut groups = BTreeMap::new();

    for global_handle in module.global_variables.iter() {
        let global = &module.global_variables[global_handle.0];
        if let Some(binding) = &global.binding {
            let group = groups.entry(binding.group).or_insert(GroupData {
                bindings: Vec::new(),
            });
            let inner_type = &module.types[module.global_variables[global_handle.0].ty].inner;

            // Assume bindings are unique since duplicates would trigger a WGSL compiler error.
            let group_binding = GroupBinding {
                binding_index: binding.binding,
                inner_type,
            };
            group.bindings.push(group_binding);
        }
    }

    groups
}
