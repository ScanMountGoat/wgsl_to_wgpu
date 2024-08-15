use crate::MatrixVectorTypes;
use naga::StructMember;
use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

pub fn shader_stages(module: &naga::Module) -> wgpu::ShaderStages {
    module
        .entry_points
        .iter()
        .map(|entry| match entry.stage {
            naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
            naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
        })
        .collect()
}

pub fn rust_scalar_type(scalar: &naga::Scalar) -> TokenStream {
    // TODO: Support other widths?
    match (scalar.kind, scalar.width) {
        (naga::ScalarKind::Sint, 1) => quote!(i8),
        (naga::ScalarKind::Uint, 1) => quote!(u8),
        (naga::ScalarKind::Sint, 2) => quote!(i16),
        (naga::ScalarKind::Uint, 2) => quote!(u16),
        (naga::ScalarKind::Sint, 4) => quote!(i32),
        (naga::ScalarKind::Uint, 4) => quote!(u32),
        (naga::ScalarKind::Float, 4) => quote!(f32),
        (naga::ScalarKind::Float, 8) => quote!(f64),
        // TODO: Do booleans have a width?
        (naga::ScalarKind::Bool, _) => quote!(bool),
        _ => todo!(),
    }
}

pub fn buffer_binding_type(storage: naga::AddressSpace) -> TokenStream {
    match storage {
        naga::AddressSpace::Uniform => quote!(wgpu::BufferBindingType::Uniform),
        naga::AddressSpace::Storage { access } => {
            let _is_read = access.contains(naga::StorageAccess::LOAD);
            let is_write = access.contains(naga::StorageAccess::STORE);

            // TODO: Is this correct?
            if is_write {
                quote!(wgpu::BufferBindingType::Storage { read_only: false })
            } else {
                quote!(wgpu::BufferBindingType::Storage { read_only: true })
            }
        }
        // This case is technically invalid.
        // Return a default to allow users to see the wgpu validation error.
        _ => quote!(wgpu::BufferBindingType::Uniform),
    }
}

pub fn rust_type(module: &naga::Module, ty: &naga::Type, format: MatrixVectorTypes) -> TokenStream {
    match &ty.inner {
        naga::TypeInner::Scalar(scalar) => rust_scalar_type(scalar),
        naga::TypeInner::Vector { size, scalar } => match format {
            MatrixVectorTypes::Rust => rust_vector_type(*size, scalar.kind, scalar.width),
            MatrixVectorTypes::Glam => glam_vector_type(*size, scalar.kind, scalar.width),
            MatrixVectorTypes::Nalgebra => nalgebra_vector_type(*size, scalar.kind, scalar.width),
        },
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => match format {
            MatrixVectorTypes::Rust => rust_matrix_type(*rows, *columns, scalar.width),
            MatrixVectorTypes::Glam => glam_matrix_type(*rows, *columns, scalar.width),
            MatrixVectorTypes::Nalgebra => nalgebra_matrix_type(*rows, *columns, scalar.width),
        },
        naga::TypeInner::Image { .. } => todo!(),
        naga::TypeInner::Sampler { .. } => todo!(),
        naga::TypeInner::Atomic(scalar) => rust_scalar_type(scalar),
        naga::TypeInner::Pointer { base: _, space: _ } => todo!(),
        naga::TypeInner::ValuePointer { .. } => todo!(),
        naga::TypeInner::Array {
            base,
            size: naga::ArraySize::Constant(size),
            stride: _,
        } => {
            let element_type = rust_type(module, &module.types[*base], format);
            let count = Literal::usize_unsuffixed(size.get() as usize);
            quote!([#element_type; #count])
        }
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => {
            panic!("Runtime-sized arrays can only be used in variable declarations or as the last field of a struct.");
        }
        naga::TypeInner::Struct {
            members: _,
            span: _,
        } => {
            let name = Ident::new(ty.name.as_ref().unwrap(), Span::call_site());
            quote!(#name)
        }
        naga::TypeInner::BindingArray { base: _, size: _ } => todo!(),
        naga::TypeInner::AccelerationStructure => todo!(),
        naga::TypeInner::RayQuery => todo!(),
    }
}

fn rust_matrix_type(rows: naga::VectorSize, columns: naga::VectorSize, width: u8) -> TokenStream {
    let inner_type = rust_scalar_type(&naga::Scalar {
        kind: naga::ScalarKind::Float,
        width,
    });
    // Use Index to generate "4" instead of "4usize".
    let rows = Literal::usize_unsuffixed(rows as usize);
    let columns = Literal::usize_unsuffixed(columns as usize);
    quote!([[#inner_type; #columns]; #rows])
}

fn glam_matrix_type(rows: naga::VectorSize, columns: naga::VectorSize, width: u8) -> TokenStream {
    // glam only supports square matrices for some types.
    // Use Rust types for unsupported matrices.
    match (rows, columns, width) {
        (naga::VectorSize::Bi, naga::VectorSize::Bi, 4) => quote!(glam::Mat2),
        (naga::VectorSize::Tri, naga::VectorSize::Tri, 4) => quote!(glam::Mat3),
        (naga::VectorSize::Quad, naga::VectorSize::Quad, 4) => quote!(glam::Mat4),
        (naga::VectorSize::Bi, naga::VectorSize::Bi, 8) => quote!(glam::DMat2),
        (naga::VectorSize::Tri, naga::VectorSize::Tri, 8) => quote!(glam::DMat3),
        (naga::VectorSize::Quad, naga::VectorSize::Quad, 8) => quote!(glam::DMat4),
        _ => rust_matrix_type(rows, columns, width),
    }
}

fn nalgebra_matrix_type(
    rows: naga::VectorSize,
    columns: naga::VectorSize,
    width: u8,
) -> TokenStream {
    let inner_type = rust_scalar_type(&naga::Scalar {
        kind: naga::ScalarKind::Float,
        width,
    });
    let rows = Literal::usize_unsuffixed(rows as usize);
    let columns = Literal::usize_unsuffixed(columns as usize);
    quote!(nalgebra::SMatrix<#inner_type, #rows, #columns>)
}

fn rust_vector_type(size: naga::VectorSize, kind: naga::ScalarKind, width: u8) -> TokenStream {
    let inner_type = rust_scalar_type(&naga::Scalar { kind, width });
    let size = Literal::usize_unsuffixed(size as usize);
    quote!([#inner_type; #size])
}

fn glam_vector_type(size: naga::VectorSize, kind: naga::ScalarKind, width: u8) -> TokenStream {
    match (size, kind, width) {
        (naga::VectorSize::Bi, naga::ScalarKind::Float, 4) => quote!(glam::Vec2),
        (naga::VectorSize::Tri, naga::ScalarKind::Float, 4) => quote!(glam::Vec3),
        (naga::VectorSize::Quad, naga::ScalarKind::Float, 4) => quote!(glam::Vec4),
        (naga::VectorSize::Bi, naga::ScalarKind::Float, 8) => quote!(glam::DVec2),
        (naga::VectorSize::Tri, naga::ScalarKind::Float, 8) => quote!(glam::DVec3),
        (naga::VectorSize::Quad, naga::ScalarKind::Float, 8) => quote!(glam::DVec4),
        (naga::VectorSize::Bi, naga::ScalarKind::Uint, 4) => quote!(glam::UVec2),
        (naga::VectorSize::Tri, naga::ScalarKind::Uint, 4) => quote!(glam::UVec3),
        (naga::VectorSize::Quad, naga::ScalarKind::Uint, 4) => quote!(glam::UVec4),
        (naga::VectorSize::Bi, naga::ScalarKind::Sint, 4) => quote!(glam::IVec2),
        (naga::VectorSize::Tri, naga::ScalarKind::Sint, 4) => quote!(glam::IVec3),
        (naga::VectorSize::Quad, naga::ScalarKind::Sint, 4) => quote!(glam::IVec4),
        // Use Rust types for unsupported types.
        _ => rust_vector_type(size, kind, width),
    }
}

fn nalgebra_vector_type(size: naga::VectorSize, kind: naga::ScalarKind, width: u8) -> TokenStream {
    let inner_type = rust_scalar_type(&naga::Scalar { kind, width });
    let size = Literal::usize_unsuffixed(size as usize);
    quote!(nalgebra::SVector<#inner_type, #size>)
}

pub fn vertex_format(ty: &naga::Type) -> wgpu::VertexFormat {
    // Not all wgsl types work as vertex attributes in wgpu.
    match &ty.inner {
        naga::TypeInner::Scalar(scalar) => match (scalar.kind, scalar.width) {
            (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32,
            (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32,
            (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32,
            (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64,
            _ => todo!(),
        },
        naga::TypeInner::Vector { size, scalar } => match size {
            naga::VectorSize::Bi => match (scalar.kind, scalar.width) {
                (naga::ScalarKind::Sint, 1) => wgpu::VertexFormat::Sint8x2,
                (naga::ScalarKind::Uint, 1) => wgpu::VertexFormat::Uint8x2,
                (naga::ScalarKind::Sint, 2) => wgpu::VertexFormat::Sint16x2,
                (naga::ScalarKind::Uint, 2) => wgpu::VertexFormat::Uint16x2,
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x2,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x2,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x2,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x2,
                _ => todo!(),
            },
            naga::VectorSize::Tri => match (scalar.kind, scalar.width) {
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x3,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x3,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x3,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x3,
                _ => todo!(),
            },
            naga::VectorSize::Quad => match (scalar.kind, scalar.width) {
                (naga::ScalarKind::Sint, 1) => wgpu::VertexFormat::Sint8x4,
                (naga::ScalarKind::Uint, 1) => wgpu::VertexFormat::Uint8x4,
                (naga::ScalarKind::Sint, 2) => wgpu::VertexFormat::Sint16x4,
                (naga::ScalarKind::Uint, 2) => wgpu::VertexFormat::Uint16x4,
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x4,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x4,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x4,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x4,
                _ => todo!(),
            },
        },
        _ => todo!(), // are these types even valid as attributes?
    }
}

#[derive(PartialEq, Eq)]
pub struct VertexInput {
    pub name: String,
    pub fields: Vec<(u32, StructMember)>,
}

// TODO: Handle errors.
// Collect the necessary data to generate an equivalent Rust struct.
pub fn get_vertex_input_structs(module: &naga::Module) -> Vec<VertexInput> {
    let mut structs: Vec<_> = module
        .entry_points
        .iter()
        .filter(|e| e.stage == naga::ShaderStage::Vertex)
        .flat_map(|vertex_entry| vertex_entry_structs(vertex_entry, module))
        .collect();

    // Remove structs that are used more than once.
    structs.sort_by_key(|s| s.name.clone());
    structs.dedup_by_key(|s| s.name.clone());

    structs
}

pub fn vertex_entry_structs(
    vertex_entry: &naga::EntryPoint,
    module: &naga::Module,
) -> Vec<VertexInput> {
    vertex_entry
        .function
        .arguments
        .iter()
        .filter(|a| a.binding.is_none())
        .filter_map(|argument| {
            let arg_type = &module.types[argument.ty];
            match &arg_type.inner {
                naga::TypeInner::Struct { members, span: _ } => {
                    let input = VertexInput {
                        name: arg_type.name.as_ref().unwrap().clone(),
                        fields: members
                            .iter()
                            .filter_map(|member| {
                                // Skip builtins since they have no location binding.
                                let location = match member.binding.as_ref().unwrap() {
                                    naga::Binding::BuiltIn(_) => None,
                                    naga::Binding::Location { location, .. } => Some(*location),
                                }?;

                                Some((location, member.clone()))
                            })
                            .collect(),
                    };

                    Some(input)
                }
                // An argument has to have a binding unless it is a structure.
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[test]
    fn shader_stages_none() {
        let source = "";
        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::NONE, shader_stages(&module));
    }

    #[test]
    fn shader_stages_vertex() {
        let source = indoc! {r#"
            @vertex
            fn main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::VERTEX, shader_stages(&module));
    }

    #[test]
    fn shader_stages_fragment() {
        let source = indoc! {r#"
            @fragment
            fn main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::FRAGMENT, shader_stages(&module));
    }

    #[test]
    fn shader_stages_vertex_fragment() {
        let source = indoc! {r#"
            @vertex
            fn vs_main()  {}

            @fragment
            fn fs_main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::VERTEX_FRAGMENT, shader_stages(&module));
    }

    #[test]
    fn shader_stages_compute() {
        let source = indoc! {r#"
            @compute
            @workgroup_size(64)
            fn main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::COMPUTE, shader_stages(&module));
    }

    #[test]
    fn shader_stages_all() {
        let source = indoc! {r#"
            @vertex
            fn vs_main()  {}

            @fragment
            fn fs_main()  {}

            @compute
            @workgroup_size(64)
            fn cs_main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::all(), shader_stages(&module));
    }
}
