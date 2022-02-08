use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, LitStr,
};
use wgsl::GroupBinding;

mod wgsl;

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
    // let wgsl_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    //     .join("src")
    //     .join(input_path.clone());
    let wgsl_path = input_path.clone();
    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();

    let module = naga::front::wgsl::parse_str(&wgsl_source).unwrap();

    let bind_group_data = wgsl::get_bind_group_data(&module);

    let bind_group_descriptors: Vec<_> = bind_group_data
        .iter()
        .map(|(group_no, _)| {
            // TODO: We've already generated these names, so it doesn't make sense to generate them again.
            let group_name = indexed_name_to_ident("BindGroup", *group_no);

            quote! {
                bind_groups::#group_name::get_bind_group_layout(&device)
            }
        })
        .collect();

    let bind_group_module = generate_bind_group_module(bind_group_data);

    let vertex_module = generate_vertex_module(&module);

    let create_pipeline_layout_function = quote! {
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
    };

    let create_shader_module_function = quote! {
        pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
            device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                // TODO: Labels?
                label: None,
                source: wgpu::ShaderSource::Wgsl(include_str!(#input_path).into()),
            })
        }
    };

    let expanded = quote! {
        // TODO: Avoid making this outer module?
        pub mod #shader_name {
            #bind_group_module
            #vertex_module

            #create_shader_module_function
            #create_pipeline_layout_function
        }
    };

    TokenStream::from(expanded)
}

fn generate_vertex_module(module: &naga::Module) -> proc_macro2::TokenStream {
    let vertex_input_locations: Vec<_> = wgsl::get_vertex_input_locations(&module)
        .iter()
        .map(|(name, location)| {
            let const_name = Ident::new(
                &format!("{}_LOCATION", name.to_uppercase()),
                Span::call_site(),
            );
            quote! {
                pub const #const_name: u32 = #location;
            }
        })
        .collect();

    quote! {
        pub mod vertex {
            #(
                #vertex_input_locations
            )*
        }
    }
}

fn generate_bind_group_module(
    bind_group_data: std::collections::BTreeMap<u32, wgsl::GroupData>,
) -> proc_macro2::TokenStream {
    let bind_group_fields = generate_bind_groups_as_fields(&bind_group_data);

    let bind_groups = generate_bind_groups(&bind_group_data);

    // TODO: How to generate setting all bind groups?
    let set_bind_groups = generate_set_bind_groups(&bind_group_data);

    quote! {
        pub mod bind_groups {
            #(
                #bind_groups
            )*

            // TODO: Add a function for setting all bind groups on a render pass?
            pub struct BindGroups<'a> {
                #(
                    pub #bind_group_fields
                ),*
            }

            // TODO: Generate this individually for each bind group type?
            pub fn set_bind_groups<'a>(render_pass: &mut wgpu::RenderPass<'a>, bind_groups: BindGroups<'a>) {
                #(#set_bind_groups)*
            }
        }
    }
}

fn generate_bind_groups_as_fields(
    bind_groups: &std::collections::BTreeMap<u32, wgsl::GroupData>,
) -> Vec<proc_macro2::TokenStream> {
    bind_groups.iter().map(|(group_no, _)| {
        let field_name = indexed_name_to_ident("bind_group", *group_no);
        let type_name = indexed_name_to_ident("BindGroup", *group_no);
        quote!{
            #field_name: &'a #type_name
        }
    }).collect()
}

fn generate_set_bind_groups(
    bind_groups: &std::collections::BTreeMap<u32, wgsl::GroupData>,
) -> Vec<proc_macro2::TokenStream> {
    bind_groups.iter().map(|(group_no, _)| {
        let field_name = indexed_name_to_ident("bind_group", *group_no);
        quote!{
            render_pass.set_bind_group(#group_no, &bind_groups.#field_name.0, &[]);
        }
    }).collect()
}

fn generate_bind_groups(
    bind_groups: &std::collections::BTreeMap<u32, wgsl::GroupData>,
) -> Vec<proc_macro2::TokenStream> {
    bind_groups.iter().map(|(group_no, group)| {
        // TODO: Creating a pipeline layout expects sets to be consecutive.
        // i.e. having groups 0 and 3 won't work but 0,1,2 will.
        let group_name = indexed_name_to_ident("BindGroup", *group_no);

        let layout_name = indexed_name_to_ident("BindGroupLayout", *group_no);
        let layout_fields: Vec<_> = group.bindings.iter().map(generate_binding_param_field).collect();

        let layout_descriptor_const_name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", *group_no);

        // TODO: Clean this up.
        // TODO: These aren't really fields anymore.
        let binding_params: Vec<_> = group.bindings.iter().map(generate_binding_param).collect();

        let descriptor_entries: Vec<_> = group.bindings.iter().map(
            generate_bind_group_layout_entry
        ).collect();

        let bind_group_entries: Vec<_> = group.bindings.iter().map(generate_bind_group_entry).collect();
        quote! {
            // The field is private to ensure the correct slot is set on the render pass.
            pub struct #group_name(wgpu::BindGroup);

            pub struct #layout_name<'a> {
                #(pub #layout_fields),*
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

            impl #group_name {
                pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                    device.create_bind_group_layout(&#layout_descriptor_const_name)
                }
                
                // TODO: This can just take a struct with all the binding params as fields instead.
                pub fn from_bindings(device: &wgpu::Device, bindings: #layout_name) -> Self {
                    // TODO: Is it possible to avoid creating the layout more than once?
                    // The wgpu types are tied to a particular device.
                    // Switching devices may invalidate the previous objects.
                    let bind_group_layout = device.create_bind_group_layout(&#layout_descriptor_const_name);
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            #(
                                #bind_group_entries
                            ),*
                        ],
                        label: None,
                    });
                    Self(bind_group)
                }

                pub fn set<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
                    render_pass.set_bind_group(#group_no, &self.0, &[]);
                }
            }
        }
    }).collect()
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{}{}", name, index), Span::call_site())
}

fn generate_bind_group_entry(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let binding_index = group_binding.binding_index;

    let binding_name = group_binding_field_name(&group_binding);

    let resource_type = match group_binding.inner_type {
        naga::TypeInner::Struct { .. } => quote! {
            // TODO: Don't assume the entire buffer is used.
            bindings.#binding_name.as_entire_binding()
        },
        naga::TypeInner::Image { .. } => quote! {
            wgpu::BindingResource::TextureView(bindings.#binding_name)
        },
        naga::TypeInner::Sampler { .. } => quote! {
            wgpu::BindingResource::Sampler(bindings.#binding_name)
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
            // TODO: Don't assume filtering?
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        },
        // TODO: Better error handling.
        _ => panic!("Failed to generate BindingType."),
    };

    quote! {
        wgpu::BindGroupLayoutEntry {
            binding: #binding_index,
            // TODO: This can't  be determined easily from WGSL, so just use both (missing compute).
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: #binding_type,
            // TODO: Support arrays?
            count: None,
        }
    }
}

// TODO: Clean up repetition.
fn generate_binding_param(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let field_name = group_binding_field_name(&group_binding);
    // TODO: Support more types.
    let field_type = match group_binding.inner_type {
        // TODO: Is it possible to make structs strongly typed and handle buffer creation automatically?
        // This could be its own module and associated tests.
        naga::TypeInner::Struct { .. } => quote! { &wgpu::Buffer },
        naga::TypeInner::Image { .. } => quote! { &wgpu::TextureView },
        naga::TypeInner::Sampler { .. } => quote! { &wgpu::Sampler },
        _ => panic!("Unsupported type for binding fields."),
    };

    quote! {
        #field_name: #field_type
    }
}

fn generate_binding_param_field(group_binding: &GroupBinding) -> proc_macro2::TokenStream {
    let field_name = group_binding_field_name(&group_binding);
    // TODO: Support more types.
    let field_type = match group_binding.inner_type {
        // TODO: Is it possible to make structs strongly typed and handle buffer creation automatically?
        // This could be its own module and associated tests.
        naga::TypeInner::Struct { .. } => quote! { &'a wgpu::Buffer },
        naga::TypeInner::Image { .. } => quote! { &'a wgpu::TextureView },
        naga::TypeInner::Sampler { .. } => quote! { &'a wgpu::Sampler },
        _ => panic!("Unsupported type for binding fields."),
    };

    quote! {
        #field_name: #field_type
    }
}


fn group_binding_field_name(binding: &GroupBinding) -> Ident {
    binding.name.as_ref().map(|name| Ident::new(name, Span::call_site())).unwrap_or_else(|| indexed_name_to_ident("binding", binding.binding_index))
}
