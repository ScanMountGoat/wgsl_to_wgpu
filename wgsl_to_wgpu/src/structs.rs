use std::collections::HashSet;

use naga::{Handle, Type};
use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::{wgsl::rust_type, TypePath, WriteOptions};

pub fn structs<F>(
    module: &naga::Module,
    options: WriteOptions,
    demangle: F,
) -> Vec<(TypePath, TokenStream)>
where
    F: Fn(&str) -> TypePath + Clone,
{
    // Initialize the layout calculator provided by naga.
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).unwrap();

    let mut global_variable_types = HashSet::new();
    for g in module.global_variables.iter() {
        add_types_recursive(&mut global_variable_types, module, g.1.ty);
    }

    // Create matching Rust structs for WGSL structs.
    // This is a UniqueArena, so each struct will only be generated once.
    module
        .types
        .iter()
        .filter(|(h, _)| {
            // Check if the struct will need to be used by the user from Rust.
            // This includes function inputs like vertex attributes and global variables.
            // Shader stage function outputs will not be accessible from Rust.
            // Skipping internal structs helps avoid issues deriving encase or bytemuck.
            !module
                .entry_points
                .iter()
                .any(|e| e.function.result.as_ref().map(|r| r.ty) == Some(*h))
                && module
                    .entry_points
                    .iter()
                    .any(|e| e.function.arguments.iter().any(|a| a.ty == *h))
                || global_variable_types.contains(h)
        })
        .filter_map(|(t_handle, t)| {
            if let naga::TypeInner::Struct { members, .. } = &t.inner {
                let path = demangle(t.name.as_ref()?);
                let s = rust_struct(
                    &path,
                    members,
                    &layouter,
                    t_handle,
                    module,
                    options,
                    &global_variable_types,
                    demangle.clone(),
                );
                Some((path, s))
            } else {
                None
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn rust_struct<F>(
    path: &TypePath,
    members: &[naga::StructMember],
    layouter: &naga::proc::Layouter,
    t_handle: naga::Handle<naga::Type>,
    module: &naga::Module,
    options: WriteOptions,
    global_variable_types: &HashSet<Handle<Type>>,
    demangle: F,
) -> TokenStream
where
    F: Fn(&str) -> TypePath + Clone,
{
    let name = &path.name;
    let struct_name = Ident::new(name, Span::call_site());

    // Skip builtins since they don't require user specified data.
    let members: Vec<_> = members
        .iter()
        .filter(|m| !matches!(m.binding, Some(naga::Binding::BuiltIn(_))))
        .cloned()
        .collect();

    let assert_member_offsets: Vec<_> = members
        .iter()
        .map(|m| {
            let field_name = Ident::new(m.name.as_ref().unwrap(), Span::call_site());
            let rust_offset = quote!(std::mem::offset_of!(#struct_name, #field_name));

            let wgsl_offset = Literal::usize_unsuffixed(m.offset as usize);

            let assert_text = format!(
                "offset of {}.{} does not match WGSL",
                name,
                m.name.as_ref().unwrap()
            );
            quote! {
                const _: () = assert!(#rust_offset == #wgsl_offset, #assert_text);
            }
        })
        .collect();

    let layout = layouter[t_handle];

    // TODO: Does the Rust alignment matter if it's copied to a buffer anyway?
    let struct_size = Literal::usize_unsuffixed(layout.size as usize);
    let assert_size_text = format!("size of {name} does not match WGSL");
    let assert_size = quote! {
        const _: () = assert!(std::mem::size_of::<#struct_name>() == #struct_size, #assert_size_text);
    };

    let has_rts_array = struct_has_rts_array_member(&members, module);
    let members = struct_members(path, &members, module, options, demangle);
    let mut derives = Vec::new();

    derives.push(quote!(Debug));
    if !has_rts_array {
        derives.push(quote!(Copy));
    }
    derives.push(quote!(Clone));
    derives.push(quote!(PartialEq));

    // Assume types used in global variables are host shareable and require validation.
    // This includes storage, uniform, and workgroup variables.
    // This also means types that are never used will not be validated.
    // Structs used only for vertex inputs do not require validation on desktop platforms.
    // Vertex input layout is handled already by setting the attribute offsets and types.
    // This allows vertex input field types without padding like vec3 for positions.
    let is_host_shareable = global_variable_types.contains(&t_handle);

    if has_rts_array && !options.derive_encase_host_shareable {
        panic!("Runtime-sized array fields are only supported with encase");
    }

    if options.derive_bytemuck_vertex && !is_host_shareable {
        if has_rts_array {
            panic!("Runtime-sized array fields are not supported with bytemuck");
        }
        derives.push(quote!(bytemuck::Pod));
        derives.push(quote!(bytemuck::Zeroable));
    }

    if options.derive_bytemuck_host_shareable && is_host_shareable {
        if has_rts_array {
            panic!("Runtime-sized array fields are not supported with bytemuck");
        }
        derives.push(quote!(bytemuck::Pod));
        derives.push(quote!(bytemuck::Zeroable));
    }

    if options.derive_encase_host_shareable && is_host_shareable {
        derives.push(quote!(encase::ShaderType));
    }

    if options.derive_serde {
        derives.push(quote!(serde::Serialize));
        derives.push(quote!(serde::Deserialize));
    }

    let assert_layout = if options.derive_bytemuck_host_shareable && is_host_shareable {
        // Assert that the Rust layout matches the WGSL layout.
        // Enable for bytemuck since it uses the Rust struct's memory layout.
        // Vertex structs have their layout manually specified and don't need validation.
        quote! {
            #assert_size
            #(#assert_member_offsets)*
        }
    } else {
        quote!()
    };

    let repr_c = if !has_rts_array {
        quote!(#[repr(C)])
    } else {
        quote!()
    };
    quote! {
        #repr_c
        #[derive(#(#derives),*)]
        pub struct #struct_name {
            #(#members),*
        }
        #assert_layout
    }
}

fn add_types_recursive(
    types: &mut HashSet<naga::Handle<naga::Type>>,
    module: &naga::Module,
    ty: Handle<Type>,
) {
    types.insert(ty);

    match &module.types[ty].inner {
        naga::TypeInner::Pointer { base, .. } => add_types_recursive(types, module, *base),
        naga::TypeInner::Array { base, .. } => add_types_recursive(types, module, *base),
        naga::TypeInner::Struct { members, .. } => {
            for member in members {
                add_types_recursive(types, module, member.ty);
            }
        }
        naga::TypeInner::BindingArray { base, .. } => add_types_recursive(types, module, *base),
        _ => (),
    }
}

fn struct_members<F>(
    path: &TypePath,
    members: &[naga::StructMember],
    module: &naga::Module,
    options: WriteOptions,
    demangle: F,
) -> Vec<TokenStream>
where
    F: Fn(&str) -> TypePath + Clone,
{
    members
        .iter()
        .enumerate()
        .map(|(index, member)| {
            let member_name = Ident::new(member.name.as_ref().unwrap(), Span::call_site());
            let ty = &module.types[member.ty];

            if let naga::TypeInner::Array {
                base,
                size: naga::ArraySize::Dynamic,
                stride: _,
            } = &ty.inner
            {
                if index != members.len() - 1 {
                    panic!("Only the last field of a struct can be a runtime-sized array");
                }
                let element_type = rust_type(
                    path,
                    module,
                    &module.types[*base],
                    options.matrix_vector_types,
                    demangle.clone(),
                );
                quote!(
                    #[size(runtime)]
                    pub #member_name: Vec<#element_type>
                )
            } else {
                let member_type = rust_type(
                    path,
                    module,
                    ty,
                    options.matrix_vector_types,
                    demangle.clone(),
                );
                quote!(pub #member_name: #member_type)
            }
        })
        .collect()
}

fn struct_has_rts_array_member(members: &[naga::StructMember], module: &naga::Module) -> bool {
    members.iter().any(|m| {
        matches!(
            module.types[m.ty].inner,
            naga::TypeInner::Array {
                size: naga::ArraySize::Dynamic,
                ..
            }
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        assert_rust_snapshot, assert_tokens_eq, create_shader_module, MatrixVectorTypes,
        ModulePath, WriteOptions,
    };
    use indoc::indoc;

    fn struct_tokens(module: &naga::Module, options: WriteOptions) -> TokenStream {
        let structs = structs(module, options, |s| TypePath {
            parent: ModulePath::default(),
            name: s.to_string(),
        });
        let structs = structs.iter().map(|(_, s)| s);
        quote!(#(#structs)*)
    }

    #[test]
    fn write_all_structs_rust() {
        let output = create_shader_module(
            include_str!("data/struct/types.wgsl"),
            "types.wgsl",
            WriteOptions::default(),
        )
        .unwrap();
        assert_rust_snapshot!(output);
    }

    #[test]
    fn write_all_structs_glam() {
        let output = create_shader_module(
            include_str!("data/struct/types.wgsl"),
            "types.wgsl",
            WriteOptions {
                matrix_vector_types: MatrixVectorTypes::Glam,
                ..Default::default()
            },
        )
        .unwrap();
        assert_rust_snapshot!(output);
    }

    #[test]
    fn write_all_structs_nalgebra() {
        let output = create_shader_module(
            include_str!("data/struct/types.wgsl"),
            "types.wgsl",
            WriteOptions {
                matrix_vector_types: MatrixVectorTypes::Nalgebra,
                ..Default::default()
            },
        )
        .unwrap();
        assert_rust_snapshot!(output);
    }

    #[test]
    fn write_all_structs_encase_bytemuck() {
        let output = create_shader_module(
            include_str!("data/struct/types.wgsl"),
            "types.wgsl",
            WriteOptions {
                derive_bytemuck_vertex: true,
                derive_bytemuck_host_shareable: true,
                derive_encase_host_shareable: true,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
                rustfmt: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_rust_snapshot!(output);
    }

    #[test]
    fn write_all_structs_serde_encase_bytemuck() {
        let output = create_shader_module(
            include_str!("data/struct/serde_encase_bytemuck.wgsl"),
            "serde_encase_bytemuck.wgsl",
            WriteOptions {
                derive_bytemuck_vertex: true,
                derive_bytemuck_host_shareable: true,
                derive_encase_host_shareable: true,
                derive_serde: true,
                matrix_vector_types: MatrixVectorTypes::Rust,
                rustfmt: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_rust_snapshot!(output);
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

            struct Unused {
                a: vec3<f32>
            }

            @fragment
            fn main(in: Input0) -> Output0 {
                var out: Output0;
                return out;
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let actual = struct_tokens(
            &module,
            WriteOptions {
                derive_bytemuck_vertex: false,
                derive_bytemuck_host_shareable: false,
                derive_encase_host_shareable: false,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
                rustfmt: true,
                ..Default::default()
            },
        );

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Input0 {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_bytemuck_skip_input_layout_validation() {
        // Structs used only for vertex inputs don't require layout validation.
        // Correctly specifying the offsets is handled by the buffer layout itself.
        let source = indoc! {r#"
            struct Input0 {
                a: u32,
                b: i32,
                c: f32,
            };

            @vertex
            fn main(input: Input0) -> vec4<f32> {
                return vec4(0.0);
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let actual = struct_tokens(
            &module,
            WriteOptions {
                derive_bytemuck_vertex: true,
                derive_bytemuck_host_shareable: true,
                derive_encase_host_shareable: false,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
                rustfmt: true,
                ..Default::default()
            },
        );

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct Input0 {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_bytemuck_input_layout_validation() {
        // The struct is also used with a storage buffer and should be validated.
        let output = create_shader_module(
            include_str!("data/struct/bytemuck_input_layout_validation.wgsl"),
            "bytemuck_input_layout_validation.wgsl",
            WriteOptions {
                derive_bytemuck_vertex: true,
                derive_bytemuck_host_shareable: true,
                derive_encase_host_shareable: false,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
                rustfmt: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_rust_snapshot!(output);
    }

    #[test]
    fn write_runtime_sized_array() {
        let source = indoc! {r#"
            struct RtsStruct {
                other_data: i32,
                the_array: array<u32>,
            };

            @group(0) @binding(0)
            var <storage, read_write> rts:RtsStruct;
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let actual = struct_tokens(
            &module,
            WriteOptions {
                derive_encase_host_shareable: true,
                ..Default::default()
            },
        );

        assert_tokens_eq!(
            quote! {
                #[derive(Debug, Clone, PartialEq, encase::ShaderType)]
                pub struct RtsStruct {
                    pub other_data: i32,
                    #[size(runtime)]
                    pub the_array: Vec<u32>,
                }
            },
            actual
        );
    }

    #[test]
    #[should_panic]
    fn write_runtime_sized_array_no_encase() {
        let source = indoc! {r#"
            struct RtsStruct {
                other_data: i32,
                the_array: array<u32>,
            };

            @group(0) @binding(0)
            var <storage, read_write> rts:RtsStruct;
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let _structs = struct_tokens(
            &module,
            WriteOptions {
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic]
    fn write_runtime_sized_array_bytemuck_vertex() {
        let source = indoc! {r#"
            struct RtsStruct {
                other_data: i32,
                the_array: array<u32>,
            };

            @vertex
            fn main(in: RtsStruct) ->  @location(0) vec4<f32> {
                return vec4(0.0);
            }
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let _structs = struct_tokens(
            &module,
            WriteOptions {
                derive_encase_host_shareable: true,
                derive_bytemuck_vertex: true,
                derive_bytemuck_host_shareable: false,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic]
    fn write_runtime_sized_array_bytemuck_host_shareable() {
        let source = indoc! {r#"
            struct RtsStruct {
                other_data: i32,
                the_array: array<u32>,
            };

            @group(0) @binding(0)
            var <storage, read_write> rts:RtsStruct;
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let _structs = struct_tokens(
            &module,
            WriteOptions {
                derive_encase_host_shareable: true,
                derive_bytemuck_vertex: false,
                derive_bytemuck_host_shareable: true,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic]
    fn write_runtime_sized_array_not_last_field() {
        let source = indoc! {r#"
            struct RtsStruct {
                other_data: i32,
                the_array: array<u32>,
                more_data: i32,
            };

            @group(0) @binding(0)
            var <storage, read_write> rts:RtsStruct;
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let _structs = struct_tokens(
            &module,
            WriteOptions {
                derive_encase_host_shareable: true,
                ..Default::default()
            },
        );
    }
}
