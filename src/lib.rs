use std::collections::{BTreeMap};

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{LitStr, parse::{Parse, ParseStream}, parse_macro_input};

struct InputData {
    shader_name: LitStr,
    path: LitStr,
}

impl Parse for InputData {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let shader_name = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let path = input.parse()?;
        Ok(InputData { shader_name, path })
    }
}

// TODO: It should be possible to configure the name and specify more than one shader file.
// Possible API: wgsl_module!("My Shader", "frag.wgsl", "vert.wgsl");
// TODO: Does wgsl support combining multiple shader files for the fragment and vertex stages?
#[proc_macro]
pub fn wgsl_module(input: TokenStream) -> TokenStream {
    // TODO: How to handle the errors from naga?
    let input = parse_macro_input!(input as InputData);
    let input_path = input.path.value();
    let shader_name = Ident::new(&input.shader_name.value(), Span::call_site());

    // TODO: This won't always use the correct path.
    // TODO: Use include_str! to recompile when the source changes?
    let wgsl_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(input_path.clone());
    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();

    let module = naga::front::wgsl::parse_str(&wgsl_source).unwrap();

    let bind_groups = get_binding_groups(&module);
    let generate_bind_groups: Vec<_> = bind_groups.iter().map(|(group_no, group)| {
        let group_name = indexed_name_to_ident("BindGroup", *group_no);
        let layout_descriptor_const_name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", *group_no);

        // TODO: Clean this up.
        let binding_fields: Vec<_> = group.bindings.iter().map(generate_binding_field).collect();

        let descriptor_entries: Vec<_> = group.bindings.iter().map(
            generate_bind_group_layout_entry
        ).collect();

        let bind_group_entries: Vec<_> = group.bindings.iter().map(generate_bind_group_entry).collect();
        
        quote! {
            pub struct #group_name<'a> {
                #(
                    #binding_fields,
                )*
            }

            const #layout_descriptor_const_name: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                // TODO: Use the name here?
                label: None,
                entries: &[
                    #(
                        #descriptor_entries
                    ),*
                ],
            };

            impl<'a> #group_name<'a> {
                pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                    device.create_bind_group_layout(&#layout_descriptor_const_name)
                }

                pub fn get_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
                    // TODO: Is it possible to avoid creating the layout more than once?
                    // The wgpu types are tied to a particular device.
                    // Switching devices may invalidate the previous objects.
                    let bind_group_layout = device.create_bind_group_layout(&#layout_descriptor_const_name);
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

    let bind_group_descriptors: Vec<_> = bind_groups.iter().map(|(group_no, _)| {
        // TODO: We've already generated these names, so it doesn't make sense to generate them again.
        let group_name = indexed_name_to_ident("BindGroup", *group_no);

        quote! {
            bind_groups::#group_name::get_bind_group_layout(&device)
        }
    }).collect();

    let expanded = quote! {
        mod #shader_name {
            pub mod bind_groups {
                #(
                    #generate_bind_groups
                )*
            }

            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    // TODO: Labels?
                    label: None,
                    flags: wgpu::ShaderFlags::all(),
                    source: wgpu::ShaderSource::Wgsl(include_str!(#input_path).into()),
                })
            }

            // TODO: Generate documentation for generated code?
            pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    // TODO: Labels?
                    label: None,
                    bind_group_layouts: &[#(&#bind_group_descriptors),*],
                    // TODO: Does wgsl have push constants?
                    push_constant_ranges: &[],
                })
            }
        }
    };

    TokenStream::from(expanded)
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{}{}", name, index), Span::call_site())
}

fn generate_bind_group_entry(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let binding_index = group_binding.binding_index;
    let binding_name = indexed_name_to_ident("binding", binding_index);
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
    let field_name = indexed_name_to_ident("binding", group_binding.binding_index);
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
