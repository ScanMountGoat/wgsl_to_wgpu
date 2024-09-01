use crate::{
    indexed_name_to_ident, quote_shader_stages, wgsl::buffer_binding_type, CreateModuleError,
};
use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use std::collections::BTreeMap;
use syn::Ident;

pub struct GroupData<'a> {
    pub bindings: Vec<GroupBinding<'a>>,
}

pub struct GroupBinding<'a> {
    pub name: Option<String>,
    pub binding_index: u32,
    pub binding_type: &'a naga::Type,
    pub address_space: naga::AddressSpace,
}

// TODO: Take an iterator instead?
pub fn bind_groups_module(
    bind_group_data: &BTreeMap<u32, GroupData>,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let bind_groups: Vec<_> = bind_group_data
        .iter()
        .map(|(group_no, group)| {
            let group_name = indexed_name_to_ident("BindGroup", *group_no);

            let layout = bind_group_layout(*group_no, group);
            let layout_descriptor = bind_group_layout_descriptor(*group_no, group, shader_stages);
            let group_impl = bind_group(*group_no, group);

            quote! {
                #[derive(Debug)]
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

    let group_parameters: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("bind_group", *group_no);
            let group_type = indexed_name_to_ident("BindGroup", *group_no);
            quote!(#group: &bind_groups::#group_type)
        })
        .collect();

    // The set function for each bind group already sets the index.
    let set_groups: Vec<_> = bind_group_data
        .keys()
        .map(|group_no| {
            let group = indexed_name_to_ident("bind_group", *group_no);
            quote!(#group.set(pass);)
        })
        .collect();

    let set_bind_groups = quote! {
        pub fn set_bind_groups<P: bind_groups::SetBindGroup>(
            pass: &mut P,
            #(#group_parameters),*
        ) {
            #(#set_groups)*
        }
    };

    if bind_groups.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        // Create a module to avoid name conflicts with user structs.
        quote! {
            pub mod bind_groups {
                #(#bind_groups)*

                #[derive(Debug, Copy, Clone)]
                pub struct BindGroups<'a> {
                    #(#bind_group_fields),*
                }

                impl<'a> BindGroups<'a> {
                    pub fn set<P: SetBindGroup>(&self, pass: &mut P) {
                        #(self.#set_groups)*
                    }
                }

                // Support both compute and render passes.
                pub trait SetBindGroup {
                    fn set_bind_group(
                        &mut self,
                        index: u32,
                        bind_group: &wgpu::BindGroup,
                        offsets: &[wgpu::DynamicOffset],
                    );
                }
                impl SetBindGroup for wgpu::ComputePass<'_> {
                    fn set_bind_group(
                        &mut self,
                        index: u32,
                        bind_group: &wgpu::BindGroup,
                        offsets: &[wgpu::DynamicOffset],
                    ) {
                        self.set_bind_group(index, bind_group, offsets);
                    }
                }
                impl SetBindGroup for wgpu::RenderPass<'_> {
                    fn set_bind_group(
                        &mut self,
                        index: u32,
                        bind_group: &wgpu::BindGroup,
                        offsets: &[wgpu::DynamicOffset],
                    ) {
                        self.set_bind_group(index, bind_group, offsets);
                    }
                }
            }
            #set_bind_groups
        }
    }
}

fn bind_group_layout(group_no: u32, group: &GroupData) -> TokenStream {
    let fields: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| {
            let binding_name = binding.name.as_ref().unwrap();
            let field_name = Ident::new(binding_name, Span::call_site());
            // TODO: Support more types.
            let field_type = match binding.binding_type.inner {
                naga::TypeInner::Struct { .. }
                | naga::TypeInner::Array { .. }
                | naga::TypeInner::Scalar { .. }
                | naga::TypeInner::Vector { .. }
                | naga::TypeInner::Matrix { .. } => quote!(wgpu::BufferBinding<'a>),
                naga::TypeInner::Image { .. } => quote!(&'a wgpu::TextureView),
                naga::TypeInner::Sampler { .. } => quote!(&'a wgpu::Sampler),
                ref inner => panic!("Unsupported type `{inner:?}` of '{binding_name}'."),
            };
            quote!(pub #field_name: #field_type)
        })
        .collect();

    let name = indexed_name_to_ident("BindGroupLayout", group_no);
    quote! {
        #[derive(Debug)]
        pub struct #name<'a> {
            #(#fields),*
        }
    }
}

fn bind_group_layout_descriptor(
    group_no: u32,
    group: &GroupData,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    let entries: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| bind_group_layout_entry(binding, shader_stages))
        .collect();

    let name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", group_no);
    let label = format!("LayoutDescriptor{group_no}");
    quote! {
        const #name: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some(#label),
            entries: &[
                #(#entries),*
            ],
        };
    }
}

fn bind_group_layout_entry(
    binding: &GroupBinding,
    shader_stages: wgpu::ShaderStages,
) -> TokenStream {
    // TODO: Assume storage is only used for compute?
    // TODO: Support just vertex or fragment?
    // TODO: Visible from all stages?
    let stages = quote_shader_stages(shader_stages);

    let binding_index = Literal::usize_unsuffixed(binding.binding_index as usize);
    let buffer_binding_type = buffer_binding_type(binding.address_space);

    // TODO: Support more types.
    let binding_type = match binding.binding_type.inner {
        naga::TypeInner::Struct { .. }
        | naga::TypeInner::Array { .. }
        | naga::TypeInner::Scalar { .. }
        | naga::TypeInner::Vector { .. }
        | naga::TypeInner::Matrix { .. } => {
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

            match class {
                naga::ImageClass::Sampled { kind, multi } => {
                    let sample_type = match kind {
                        naga::ScalarKind::Sint => quote!(wgpu::TextureSampleType::Sint),
                        naga::ScalarKind::Uint => quote!(wgpu::TextureSampleType::Uint),
                        naga::ScalarKind::Float => {
                            // TODO: Don't assume all textures are filterable.
                            quote!(wgpu::TextureSampleType::Float { filterable: true })
                        }
                        _ => todo!(),
                    };
                    quote!(wgpu::BindingType::Texture {
                        sample_type: #sample_type,
                        view_dimension: #view_dim,
                        multisampled: #multi,
                    })
                }
                naga::ImageClass::Depth { multi } => {
                    quote!(wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: #view_dim,
                        multisampled: #multi,
                    })
                }
                naga::ImageClass::Storage { format, access } => {
                    // TODO: Will the debug implementation always work with the macro?
                    // Assume texture format variants are the same as storage formats.
                    let format = syn::Ident::new(&format!("{format:?}"), Span::call_site());
                    let storage_access = storage_access(access);

                    quote!(wgpu::BindingType::StorageTexture {
                        access: #storage_access,
                        format: wgpu::TextureFormat::#format,
                        view_dimension: #view_dim,
                    })
                }
            }
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
        ref inner => {
            panic!("Failed to generate BindingType for `{inner:?}` at index {binding_index}.",)
        }
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

fn storage_access(access: naga::StorageAccess) -> TokenStream {
    let is_read = access.contains(naga::StorageAccess::LOAD);
    let is_write = access.contains(naga::StorageAccess::STORE);
    match (is_read, is_write) {
        (true, true) => quote!(wgpu::StorageTextureAccess::ReadWrite),
        (true, false) => quote!(wgpu::StorageTextureAccess::ReadOnly),
        (false, true) => quote!(wgpu::StorageTextureAccess::WriteOnly),
        _ => todo!(), // shouldn't be possible
    }
}

fn bind_group(group_no: u32, group: &GroupData) -> TokenStream {
    let entries: Vec<_> = group
        .bindings
        .iter()
        .map(|binding| {
            let binding_index = Literal::usize_unsuffixed(binding.binding_index as usize);
            let binding_name = binding.name.as_ref().unwrap();
            let field_name = Ident::new(binding.name.as_ref().unwrap(), Span::call_site());
            let resource_type = match binding.binding_type.inner {
                naga::TypeInner::Struct { .. }
                | naga::TypeInner::Array { .. }
                | naga::TypeInner::Scalar { .. }
                | naga::TypeInner::Vector { .. }
                | naga::TypeInner::Matrix { .. } => {
                    quote!(wgpu::BindingResource::Buffer(bindings.#field_name))
                }
                naga::TypeInner::Image { .. } => {
                    quote!(wgpu::BindingResource::TextureView(bindings.#field_name))
                }
                naga::TypeInner::Sampler { .. } => {
                    quote!(wgpu::BindingResource::Sampler(bindings.#field_name))
                }
                // TODO: Better error handling.
                ref inner => panic!(
                    "Failed to generate BindingType for `{inner:?}` of '{binding_name}' at index {binding_index}.",
                ),
            };

            quote! {
                wgpu::BindGroupEntry {
                    binding: #binding_index,
                    resource: #resource_type,
                }
            }
        })
        .collect();

    let bind_group_name = indexed_name_to_ident("BindGroup", group_no);
    let bind_group_layout_name = indexed_name_to_ident("BindGroupLayout", group_no);

    let layout_descriptor_name = indexed_name_to_ident("LAYOUT_DESCRIPTOR", group_no);

    let label = format!("BindGroup{group_no}");

    let group_no = Literal::usize_unsuffixed(group_no as usize);

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
                    label: Some(#label),
                });
                Self(bind_group)
            }

            pub fn set<P: SetBindGroup>(&self, pass: &mut P) {
                pass.set_bind_group(#group_no, &self.0, &[]);
            }
        }
    }
}

pub fn get_bind_group_data(
    module: &naga::Module,
) -> Result<BTreeMap<u32, GroupData>, CreateModuleError> {
    // Use a BTree to sort type and field names by group index.
    // This isn't strictly necessary but makes the generated code cleaner.
    let mut groups = BTreeMap::new();

    for global_handle in module.global_variables.iter() {
        let global = &module.global_variables[global_handle.0];
        if let Some(binding) = &global.binding {
            let group = groups.entry(binding.group).or_insert(GroupData {
                bindings: Vec::new(),
            });
            let binding_type = &module.types[module.global_variables[global_handle.0].ty];

            let group_binding = GroupBinding {
                name: global.name.clone(),
                binding_index: binding.binding,
                binding_type,
                address_space: global.space,
            };
            // Repeated bindings will probably cause a compile error.
            // We'll still check for it here just in case.
            if group
                .bindings
                .iter()
                .any(|g| g.binding_index == binding.binding)
            {
                return Err(CreateModuleError::DuplicateBinding {
                    binding: binding.binding,
                });
            }
            group.bindings.push(group_binding);
        }
    }

    // wgpu expects bind groups to be consecutive starting from 0.
    if groups.keys().map(|i| *i as usize).eq(0..groups.len()) {
        Ok(groups)
    } else {
        Err(CreateModuleError::NonConsecutiveBindGroups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tokens_eq;
    use indoc::indoc;

    #[test]
    fn bind_group_data_consecutive_bind_groups() {
        let source = indoc! {r#"
            @group(0) @binding(0) var<uniform> a: vec4<f32>;
            @group(1) @binding(0) var<uniform> b: vec4<f32>;
            @group(2) @binding(0) var<uniform> c: vec4<f32>;

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(3, get_bind_group_data(&module).unwrap().len());
    }

    #[test]
    fn bind_group_data_first_group_not_zero() {
        let source = indoc! {r#"
            @group(1) @binding(0) var<uniform> a: vec4<f32>;

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert!(matches!(
            get_bind_group_data(&module),
            Err(CreateModuleError::NonConsecutiveBindGroups)
        ));
    }

    #[test]
    fn bind_group_data_non_consecutive_bind_groups() {
        let source = indoc! {r#"
            @group(0) @binding(0) var<uniform> a: vec4<f32>;
            @group(1) @binding(0) var<uniform> b: vec4<f32>;
            @group(3) @binding(0) var<uniform> c: vec4<f32>;

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert!(matches!(
            get_bind_group_data(&module),
            Err(CreateModuleError::NonConsecutiveBindGroups)
        ));
    }

    fn test_bind_groups(wgsl: &str, rust: &str, stages: wgpu::ShaderStages) {
        let module = naga::front::wgsl::parse_str(wgsl).unwrap();
        let bind_group_data = get_bind_group_data(&module).unwrap();
        let actual = bind_groups_module(&bind_group_data, stages);

        assert_tokens_eq!(rust.parse().unwrap(), actual);
    }

    #[test]
    fn bind_groups_module_compute() {
        test_bind_groups(
            include_str!("data/bindgroup/compute.wgsl"),
            include_str!("data/bindgroup/compute.rs"),
            wgpu::ShaderStages::COMPUTE,
        );
    }

    #[test]
    fn bind_groups_module_vertex_fragment() {
        // Test different texture and sampler types.
        // TODO: Storage textures.
        test_bind_groups(
            include_str!("data/bindgroup/vertex_fragment.wgsl"),
            include_str!("data/bindgroup/vertex_fragment.rs"),
            wgpu::ShaderStages::VERTEX_FRAGMENT,
        );
    }

    #[test]
    fn bind_groups_module_vertex() {
        // The actual content of the structs doesn't matter.
        // We only care about the groups and bindings.
        test_bind_groups(
            include_str!("data/bindgroup/vertex.wgsl"),
            include_str!("data/bindgroup/vertex.rs"),
            wgpu::ShaderStages::VERTEX,
        );
    }

    #[test]
    fn bind_groups_module_fragment() {
        // The actual content of the structs doesn't matter.
        // We only care about the groups and bindings.
        test_bind_groups(
            include_str!("data/bindgroup/fragment.wgsl"),
            include_str!("data/bindgroup/fragment.rs"),
            wgpu::ShaderStages::FRAGMENT,
        );
    }
}
