use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Index};

use crate::{wgsl::rust_type, MatrixVectorTypes, WriteOptions};

pub fn structs(module: &naga::Module, options: WriteOptions) -> Vec<TokenStream> {
    // Initialize the layout calculator provided by naga.
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(&module.types, &module.constants).unwrap();

    // Create matching Rust structs for WGSL structs.
    // This is a UniqueArena, so each struct will only be generated once.
    module
        .types
        .iter()
        .filter(|(h, _)| {
            // Shader stage outputs don't need to be instantiated by the user.
            // Many builtin outputs also don't satisfy alignment requirements.
            // Skipping these structs helps avoid issues deriving encase or bytemuck.
            !module
                .entry_points
                .iter()
                .any(|e| e.function.result.as_ref().map(|r| r.ty) == Some(*h))
        })
        .filter_map(|(t_handle, t)| {
            if let naga::TypeInner::Struct { members, .. } = &t.inner {
                let struct_name = Ident::new(t.name.as_ref().unwrap(), Span::call_site());

                let assert_member_offsets: Vec<_> = members
                    .iter()
                    .map(|m| {
                        let name = Ident::new(m.name.as_ref().unwrap(), Span::call_site());
                        let rust_offset = quote!(memoffset::offset_of!(#struct_name, #name));

                        let wgsl_offset = Index::from(m.offset as usize);

                        let assert_text = format!("offset of {}.{} does not match WGSL", t.name.as_ref().unwrap(), m.name.as_ref().unwrap());
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
                if options.derive_serde {
                    derives.push(quote!(serde::Serialize));
                    derives.push(quote!(serde::Deserialize));
                }

                let assert_layout = if options.derive_bytemuck {
                    // Assert that the Rust layout matches the WGSL layout.
                    // Enable for bytemuck since it uses the Rust struct's memory layout.
                    // TODO: This isn't necessary for vertex input structs?
                    // TODO: Add an option to still validate vertex inputs for storage buffer compatibility.
                    quote!{
                        #assert_size
                        #(#assert_member_offsets)*
                    }
                } else {
                    quote!()
                };

                Some(quote! {
                    #[repr(C)]
                    #[derive(#(#derives),*)]
                    pub struct #struct_name {
                        #(#members),*
                    }
                    #assert_layout
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
            let member_type = rust_type(module, &module.types[member.ty], format);
            quote!(pub #member_name: #member_type)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{assert_tokens_eq, WriteOptions};
    use indoc::indoc;

    #[test]
    fn write_all_structs_rust() {
        let source = indoc! {r#"
            struct Scalars {
                a: u32,
                b: i32,
                c: f32,
            };

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
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

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
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

            struct VectorsU32 {
                a: vec2<u32>,
                b: vec3<u32>,
                c: vec4<u32>,
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

            @fragment
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
}
