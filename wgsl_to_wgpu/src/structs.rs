use std::collections::HashSet;

use naga::{Handle, Type};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Index};

use crate::{wgsl::rust_type, WriteOptions};

pub fn structs(module: &naga::Module, options: WriteOptions) -> Vec<TokenStream> {
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
                Some(rust_struct(
                    t,
                    members,
                    &layouter,
                    t_handle,
                    module,
                    options,
                    &global_variable_types,
                ))
            } else {
                None
            }
        })
        .collect()
}

fn rust_struct(
    t: &naga::Type,
    members: &[naga::StructMember],
    layouter: &naga::proc::Layouter,
    t_handle: naga::Handle<naga::Type>,
    module: &naga::Module,
    options: WriteOptions,
    global_variable_types: &HashSet<Handle<Type>>,
) -> TokenStream {
    let struct_name = Ident::new(t.name.as_ref().unwrap(), Span::call_site());

    let assert_member_offsets: Vec<_> = members
        .iter()
        .map(|m| {
            let name = Ident::new(m.name.as_ref().unwrap(), Span::call_site());
            let rust_offset = quote!(memoffset::offset_of!(#struct_name, #name));

            let wgsl_offset = Index::from(m.offset as usize);

            let assert_text = format!(
                "offset of {}.{} does not match WGSL",
                t.name.as_ref().unwrap(),
                m.name.as_ref().unwrap()
            );
            quote! {
                const _: () = assert!(#rust_offset == #wgsl_offset, #assert_text);
            }
        })
        .collect();

    let layout = layouter[t_handle];

    // TODO: Does the Rust alignment matter if it's copied to a buffer anyway?
    let struct_size = Index::from(layout.size as usize);
    let assert_size_text = format!("size of {} does not match WGSL", t.name.as_ref().unwrap());
    let assert_size = quote! {
        const _: () = assert!(std::mem::size_of::<#struct_name>() == #struct_size, #assert_size_text);
    };

    // Assume types used in global variables are host shareable and require validation.
    // This includes storage, uniform, and workgroup variables.
    // This also means types that are never used will not be validated.
    // Structs used only for vertex inputs do not require validation on desktop platforms.
    // Vertex input layout is handled already by setting the attribute offsets and types.
    // This allows vertex input field types without padding like vec3 for positions.
    let is_host_shareable = global_variable_types.contains(&t_handle);

    let has_rts_array = struct_has_rts_array_member(members, module);
    let members = struct_members(members, module, options, layout.size as usize, is_host_shareable);
    let mut derives = Vec::new();

    derives.push(quote!(Debug));
    if !has_rts_array {
        derives.push(quote!(Copy));
    }
    derives.push(quote!(Clone));
    derives.push(quote!(PartialEq));

    if options.derive_bytemuck {
        if has_rts_array {
            panic!("Runtime-sized array fields are not supported in options.derive_bytemuck mode");
        }
        derives.push(quote!(bytemuck::Pod));
        derives.push(quote!(bytemuck::Zeroable));
    }
    if options.derive_encase {
        derives.push(quote!(encase::ShaderType));
    } else if has_rts_array {
        panic!("Runtime-sized array fields are only supported in options.derive_encase mode");
    }
    if options.derive_serde {
        derives.push(quote!(serde::Serialize));
        derives.push(quote!(serde::Deserialize));
    }

    let assert_layout = if options.derive_bytemuck && is_host_shareable {
        // Assert that the Rust layout matches the WGSL layout.
        // Enable for bytemuck since it uses the Rust struct's memory layout.
        quote! {
            #assert_size
            #(#assert_member_offsets)*
        }
    } else {
        quote!()
    };

    let repr_c = if !has_rts_array {
        quote!(#[repr(C, packed)])
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

fn struct_members(
    members: &[naga::StructMember],
    module: &naga::Module,
    options: WriteOptions,
    struct_size: usize, 
    enable_padding: bool,
) -> Vec<TokenStream> {
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
                let element_type =
                    rust_type(module, &module.types[*base], options.matrix_vector_types);
                quote!(
                    #[size(runtime)]
                    pub #member_name: Vec<#element_type>
                )
            } else {
              let member_type = rust_type(module, ty, options.matrix_vector_types);
              if !enable_padding {
                quote!(pub #member_name: #member_type)
              } else {
                let current_offset = Index::from(member.offset as usize);

                let next_offset = if index == members.len() - 1 {
                  Index::from(struct_size)
                } else {
                  Index::from(members[index + 1].offset as usize)
                };

                let pad_member_name = Ident::new(&format!("_pad_{}", member_name), Span::call_site());

                quote!(
                  pub #member_name: #member_type,
                  pub #pad_member_name: [u8; #next_offset - #current_offset - core::mem::size_of::<#member_type>()]
                )
              }
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
    use indoc::indoc;

    use super::*;
    use crate::{assert_tokens_eq, MatrixVectorTypes, WriteOptions};

    #[test]
    fn write_all_structs_rust() {
        let source = indoc! {r#"
            struct Scalars {
                a: u32,
                b: i32,
                c: f32,
            };
            var<uniform> a: Scalars;

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
            };
            var<uniform> b: VectorsU32;

            struct VectorsI32 {
                a: vec2<i32>,
                b: vec3<i32>,
                c: vec4<i32>,
            };
            var<uniform> c: VectorsI32;

            struct VectorsF32 {
                a: vec2<f32>,
                b: vec3<f32>,
                c: vec4<f32>,
            };
            var<uniform> d: VectorsF32;

            struct VectorsF64 {
                a: vec2<f64>,
                b: vec3<f64>,
                c: vec4<f64>,
            };
            var<uniform> e: VectorsF64;

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
            var<uniform> f: MatricesF32;

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
            var<uniform> g: MatricesF64;

            struct StaticArrays {
                a: array<u32, 5>,
                b: array<f32, 3>,
                c: array<mat4x4<f32>, 512>,
            };
            var<uniform> h: StaticArrays;

            struct Nested {
                a: MatricesF32,
                b: MatricesF64
            }
            var<uniform> i: Nested;

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(&module, WriteOptions::default());
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Scalars {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
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
            var<uniform> a: Scalars;

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
            };
            var<uniform> b: VectorsU32;

            struct VectorsI32 {
                a: vec2<i32>,
                b: vec3<i32>,
                c: vec4<i32>,
            };
            var<uniform> c: VectorsI32;

            struct VectorsF32 {
                a: vec2<f32>,
                b: vec3<f32>,
                c: vec4<f32>,
            };
            var<uniform> d: VectorsF32;

            struct VectorsF64 {
                a: vec2<f64>,
                b: vec3<f64>,
                c: vec4<f64>,
            };
            var<uniform> e: VectorsF64;

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
            var<uniform> f: MatricesF32;

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
            var<uniform> g: MatricesF64;

            struct StaticArrays {
                a: array<u32, 5>,
                b: array<f32, 3>,
                c: array<mat4x4<f32>, 512>,
            };
            var<uniform> h: StaticArrays;

            struct Nested {
                a: MatricesF32,
                b: MatricesF64
            }
            var<uniform> i: Nested;

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
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Scalars {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
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
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_nalgebra() {
        let source = indoc! {r#"
            struct Scalars {
                a: u32,
                b: i32,
                c: f32,
            };
            var<uniform> a: Scalars;

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
            };
            var<uniform> b: VectorsU32;

            struct VectorsI32 {
                a: vec2<i32>,
                b: vec3<i32>,
                c: vec4<i32>,
            };
            var<uniform> c: VectorsI32;

            struct VectorsF32 {
                a: vec2<f32>,
                b: vec3<f32>,
                c: vec4<f32>,
            };
            var<uniform> d: VectorsF32;

            struct VectorsF64 {
                a: vec2<f64>,
                b: vec3<f64>,
                c: vec4<f64>,
            };
            var<uniform> e: VectorsF64;

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
            var<uniform> f: MatricesF32;

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
            var<uniform> g: MatricesF64;

            struct StaticArrays {
                a: array<u32, 5>,
                b: array<f32, 3>,
                c: array<mat4x4<f32>, 512>,
            };
            var<uniform> h: StaticArrays;

            struct Nested {
                a: MatricesF32,
                b: MatricesF64
            }
            var<uniform> i: Nested;

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                matrix_vector_types: MatrixVectorTypes::Nalgebra,
                ..Default::default()
            },
        );
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Scalars {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsU32 {
                    pub a: nalgebra::SVector<u32, 2>,
                    pub b: nalgebra::SVector<u32, 3>,
                    pub c: nalgebra::SVector<u32, 4>,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsI32 {
                    pub a: nalgebra::SVector<i32, 2>,
                    pub b: nalgebra::SVector<i32, 3>,
                    pub c: nalgebra::SVector<i32, 4>,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsF32 {
                    pub a: nalgebra::SVector<f32, 2>,
                    pub b: nalgebra::SVector<f32, 3>,
                    pub c: nalgebra::SVector<f32, 4>,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct VectorsF64 {
                    pub a: nalgebra::SVector<f64, 2>,
                    pub b: nalgebra::SVector<f64, 3>,
                    pub c: nalgebra::SVector<f64, 4>,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct MatricesF32 {
                    pub a: nalgebra::SMatrix<f32, 4, 4>,
                    pub b: nalgebra::SMatrix<f32, 3, 4>,
                    pub c: nalgebra::SMatrix<f32, 2, 4>,
                    pub d: nalgebra::SMatrix<f32, 4, 3>,
                    pub e: nalgebra::SMatrix<f32, 3, 3>,
                    pub f: nalgebra::SMatrix<f32, 2, 3>,
                    pub g: nalgebra::SMatrix<f32, 4, 2>,
                    pub h: nalgebra::SMatrix<f32, 3, 2>,
                    pub i: nalgebra::SMatrix<f32, 2, 2>,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct MatricesF64 {
                    pub a: nalgebra::SMatrix<f64, 4, 4>,
                    pub b: nalgebra::SMatrix<f64, 3, 4>,
                    pub c: nalgebra::SMatrix<f64, 2, 4>,
                    pub d: nalgebra::SMatrix<f64, 4, 3>,
                    pub e: nalgebra::SMatrix<f64, 3, 3>,
                    pub f: nalgebra::SMatrix<f64, 2, 3>,
                    pub g: nalgebra::SMatrix<f64, 4, 2>,
                    pub h: nalgebra::SMatrix<f64, 3, 2>,
                    pub i: nalgebra::SMatrix<f64, 2, 2>,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct StaticArrays {
                    pub a: [u32; 5],
                    pub b: [f32; 3],
                    pub c: [nalgebra::SMatrix<f32, 4, 4>; 512],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Nested {
                    pub a: MatricesF32,
                    pub b: MatricesF64,
                }
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

            struct Nested {
                a: Input0,
                b: f32
            }

            var<uniform> a: Input0;
            var<storage, read> b: Nested;

            @fragment
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: true,
                derive_bytemuck: true,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
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
                const _: () = assert!(
                    std::mem::size_of:: < Input0 > () == 12, "size of Input0 does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, a) == 0, "offset of Input0.a does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, b) == 4, "offset of Input0.b does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, c) == 8, "offset of Input0.c does not match WGSL"
                );
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
                pub struct Nested {
                    pub a: Input0,
                    pub b: f32,
                }
                const _: () = assert!(
                    std::mem::size_of:: < Nested > () == 16, "size of Nested does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Nested, a) == 0, "offset of Nested.a does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Nested, b) == 12, "offset of Nested.b does not match WGSL"
                );
            },
            actual
        );
    }

    #[test]
    fn write_all_structs_serde_encase_bytemuck() {
        let source = indoc! {r#"
            struct Input0 {
                a: u32,
                b: i32,
                c: f32,
            };

            struct Nested {
                a: Input0,
                b: f32
            }

            var<workgroup> a: Input0;
            var<uniform> b: Nested;

            @compute
            @workgroup_size(64)
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: true,
                derive_bytemuck: true,
                derive_serde: true,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(
                    Debug,
                    Copy,
                    Clone,
                    PartialEq,
                    bytemuck::Pod,
                    bytemuck::Zeroable,
                    encase::ShaderType,
                    serde::Serialize,
                    serde::Deserialize
                )]
                pub struct Input0 {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                const _: () = assert!(
                    std::mem::size_of:: < Input0 > () == 12, "size of Input0 does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, a) == 0, "offset of Input0.a does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, b) == 4, "offset of Input0.b does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, c) == 8, "offset of Input0.c does not match WGSL"
                );
                #[repr(C)]
                #[derive(
                    Debug,
                    Copy,
                    Clone,
                    PartialEq,
                    bytemuck::Pod,
                    bytemuck::Zeroable,
                    encase::ShaderType,
                    serde::Serialize,
                    serde::Deserialize
                )]
                pub struct Nested {
                    pub a: Input0,
                    pub b: f32,
                }
                const _: () = assert!(
                    std::mem::size_of:: < Nested > () == 16, "size of Nested does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Nested, a) == 0, "offset of Nested.a does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Nested, b) == 12, "offset of Nested.b does not match WGSL"
                );
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

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: false,
                derive_bytemuck: false,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = quote!(#(#structs)*);

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

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: false,
                derive_bytemuck: true,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = quote!(#(#structs)*);

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
        let source = indoc! {r#"
            struct Input0 {
                @size(8)
                a: u32,
                b: i32,
                @align(32)
                c: f32,
            };

            var<storage, read_write> test: Input0;

            struct Outer {
                inner: Inner
            }

            struct Inner {
                a: f32
            }

            var<storage, read_write> test2: array<Outer>;

            @vertex
            fn main(input: Input0) -> vec4<f32> {
                return vec4(0.0);
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: false,
                derive_bytemuck: true,
                derive_serde: false,
                matrix_vector_types: MatrixVectorTypes::Rust,
            },
        );
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct Input0 {
                    pub a: u32,
                    pub b: i32,
                    pub c: f32,
                }
                const _: () = assert!(
                    std::mem::size_of:: < Input0 > () == 64, "size of Input0 does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, a) == 0, "offset of Input0.a does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, b) == 8, "offset of Input0.b does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Input0, c) == 32, "offset of Input0.c does not match WGSL"
                );
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct Inner {
                    pub a: f32,
                }
                const _: () = assert!(
                    std::mem::size_of:: < Inner > () == 4, "size of Inner does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Inner, a) == 0, "offset of Inner.a does not match WGSL"
                );
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct Outer {
                    pub inner: Inner,
                }
                const _: () = assert!(
                    std::mem::size_of:: < Outer > () == 4, "size of Outer does not match WGSL"
                );
                const _: () = assert!(
                    memoffset::offset_of!(Outer, inner) == 0, "offset of Outer.inner does not match WGSL"
                );
            },
            actual
        );
    }

    #[test]
    fn write_atomic_types() {
        let source = indoc! {r#"
            struct Atomics {
                num: atomic<u32>,
                numi: atomic<i32>,
            };

            @group(0) @binding(0)
            var <storage, read_write> atomics:Atomics;
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let structs = structs(
            &module,
            WriteOptions {
                matrix_vector_types: MatrixVectorTypes::Nalgebra,
                ..Default::default()
            },
        );
        let actual = quote!(#(#structs)*);

        assert_tokens_eq!(
            quote! {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct Atomics {
                    pub num: u32,
                    pub numi: i32,
                }
            },
            actual
        );
    }

    fn runtime_sized_array_module() -> naga::Module {
        let source = indoc! {r#"
            struct RtsStruct {
                other_data: i32,
                the_array: array<u32>,
            };

            @group(0) @binding(0)
            var <storage, read_write> rts:RtsStruct;
        "#};
        naga::front::wgsl::parse_str(source).unwrap()
    }

    #[test]
    fn write_runtime_sized_array() {
        let module = runtime_sized_array_module();

        let structs = structs(
            &module,
            WriteOptions {
                derive_encase: true,
                ..Default::default()
            },
        );
        let actual = quote!(#(#structs)*);

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
        let module = runtime_sized_array_module();

        let _structs = structs(
            &module,
            WriteOptions {
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic]
    fn write_runtime_sized_array_bytemuck() {
        let module = runtime_sized_array_module();

        let _structs = structs(
            &module,
            WriteOptions {
                derive_encase: true,
                derive_bytemuck: true,
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

        let _structs = structs(
            &module,
            WriteOptions {
                derive_encase: true,
                ..Default::default()
            },
        );
    }
}
