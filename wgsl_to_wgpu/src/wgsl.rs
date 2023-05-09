use crate::MatrixVectorTypes;
use naga::StructMember;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Index};

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

fn rust_scalar_type(kind: naga::ScalarKind, width: u8) -> TokenStream {
    // TODO: Support other widths?
    match (kind, width) {
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
        _ => todo!(),
    }
}

pub fn rust_type(module: &naga::Module, ty: &naga::Type, format: MatrixVectorTypes) -> TokenStream {
    match &ty.inner {
        naga::TypeInner::Scalar { kind, width } => rust_scalar_type(*kind, *width),
        naga::TypeInner::Vector { size, kind, width } => match format {
            MatrixVectorTypes::Rust => rust_vector_type(*size, *kind, *width),
            MatrixVectorTypes::Glam => glam_vector_type(*size, *kind, *width),
            MatrixVectorTypes::Nalgebra => nalgebra_vector_type(*size, *kind, *width),
        },
        naga::TypeInner::Matrix {
            columns,
            rows,
            width,
        } => match format {
            MatrixVectorTypes::Rust => rust_matrix_type(*rows, *columns, *width),
            MatrixVectorTypes::Glam => glam_matrix_type(*rows, *columns, *width),
            MatrixVectorTypes::Nalgebra => nalgebra_matrix_type(*rows, *columns, *width),
        },
        naga::TypeInner::Image { .. } => todo!(),
        naga::TypeInner::Sampler { .. } => todo!(),
        naga::TypeInner::Atomic { kind, width } => rust_scalar_type(*kind, *width),
        naga::TypeInner::Pointer { base: _, space: _ } => todo!(),
        naga::TypeInner::ValuePointer {
            size: _,
            kind: _,
            width: _,
            space: _,
        } => todo!(),
        naga::TypeInner::Array {
            base,
            size,
            stride: _,
        } => {
            // TODO: Support arrays other than arrays with a static size?
            let element_type = rust_type(module, &module.types[*base], format);
            let count = Index::from(array_length(size, module));
            quote!([#element_type; #count])
        }
        naga::TypeInner::Struct {
            members: _,
            span: _,
        } => {
            // TODO: Support structs?
            let name = Ident::new(ty.name.as_ref().unwrap(), Span::call_site());
            quote!(#name)
        }
        naga::TypeInner::BindingArray { base: _, size: _ } => todo!(),
        naga::TypeInner::AccelerationStructure => todo!(),
        naga::TypeInner::RayQuery => todo!(),
    }
}

fn rust_matrix_type(rows: naga::VectorSize, columns: naga::VectorSize, width: u8) -> TokenStream {
    let inner_type = rust_scalar_type(naga::ScalarKind::Float, width);
    // Use Index to generate "4" instead of "4usize".
    let rows = Index::from(rows as usize);
    let columns = Index::from(columns as usize);
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
    let inner_type = rust_scalar_type(naga::ScalarKind::Float, width);
    let rows = Index::from(rows as usize);
    let columns = Index::from(columns as usize);
    quote!(nalgebra::SMatrix<#inner_type, #rows, #columns>)
}

fn rust_vector_type(size: naga::VectorSize, kind: naga::ScalarKind, width: u8) -> TokenStream {
    let inner_type = rust_scalar_type(kind, width);
    let size = Index::from(size as usize);
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
    let inner_type = rust_scalar_type(kind, width);
    let size = Index::from(size as usize);
    quote!(nalgebra::SVector<#inner_type, #size>)
}

pub fn vertex_format(ty: &naga::Type) -> wgpu::VertexFormat {
    // Not all wgsl types work as vertex attributes in wgpu.
    match &ty.inner {
        naga::TypeInner::Scalar { kind, width } => match (kind, width) {
            (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32,
            (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32,
            (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32,
            (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64,
            _ => todo!(),
        },
        naga::TypeInner::Vector { size, kind, width } => match size {
            naga::VectorSize::Bi => match (kind, width) {
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
            naga::VectorSize::Tri => match (kind, width) {
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x3,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x3,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x3,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x3,
                _ => todo!(),
            },
            naga::VectorSize::Quad => match (kind, width) {
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

fn array_length(size: &naga::ArraySize, module: &naga::Module) -> usize {
    match size {
        naga::ArraySize::Constant(c) => match &module.constants[*c].inner {
            naga::ConstantInner::Scalar { value, .. } => match value {
                naga::ScalarValue::Sint(v) => *v as usize,
                naga::ScalarValue::Uint(v) => *v as usize,
                naga::ScalarValue::Float(_) => todo!(),
                naga::ScalarValue::Bool(_) => todo!(),
            },
            _ => todo!(),
        },
        naga::ArraySize::Dynamic => 0, // TODO: how to handle dynamically sized arrays?
    }
}

pub struct VertexInput {
    pub name: String,
    pub fields: Vec<(u32, StructMember)>,
}

// TODO: Handle errors.
// Collect the necessary data to generate an equivalent Rust struct.
pub fn get_vertex_input_structs(module: &naga::Module) -> Vec<VertexInput> {
    // TODO: Handle multiple entries?
    module
        .entry_points
        .iter()
        .find(|e| e.stage == naga::ShaderStage::Vertex)
        .map(|vertex_entry| {
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
                                    .map(|member| {
                                        let location = match member.binding.as_ref().unwrap() {
                                            naga::Binding::BuiltIn(_) => todo!(), // TODO: is it possible to have builtins for inputs?
                                            naga::Binding::Location { location, .. } => *location,
                                        };

                                        (location, member.clone())
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
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[test]
    fn shader_stages_none() {
        let source = indoc! {r#"

        "#};

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
            fn cs_main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::all(), shader_stages(&module));
    }

    #[test]
    fn vertex_input_structs_two_structs() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) in0: vec4<f32>,
                @location(1) in1: vec4<f32>,
                @location(2) in2: vec4<f32>,
            };
            
            struct VertexInput1 {
                @location(3) in3: vec4<f32>,
                @location(4) in4: vec4<f32>,
                @location(5) in5: vec4<f32>,
                @location(6) in6: vec4<u32>,
            };

            @vertex
            fn main(
                in0: VertexInput0,
                in1: VertexInput1,
                @builtin(front_facing) in2: bool,
                @location(7) in3: vec4<f32>,
            ) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let vertex_inputs = get_vertex_input_structs(&module);
        // Only structures should be included.
        assert_eq!(2, vertex_inputs.len());

        assert_eq!("VertexInput0", vertex_inputs[0].name);
        assert_eq!(3, vertex_inputs[0].fields.len());
        assert_eq!("in1", vertex_inputs[0].fields[1].1.name.as_ref().unwrap());
        assert_eq!(1, vertex_inputs[0].fields[1].0);

        assert_eq!("VertexInput1", vertex_inputs[1].name);
        assert_eq!(4, vertex_inputs[1].fields.len());
        assert_eq!("in5", vertex_inputs[1].fields[2].1.name.as_ref().unwrap());
        assert_eq!(5, vertex_inputs[1].fields[2].0);
    }
}
