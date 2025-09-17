use std::{collections::BTreeMap, path::PathBuf};

use crate::{MatrixVectorTypes, TypePath};
use naga::StructMember;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

pub fn global_shader_stages(module: &naga::Module) -> BTreeMap<String, wgpu::ShaderStages> {
    // Collect the shader stages for all entries that access a global variable.
    // This is referred to as being "statically accessed" in the WGSL specification.
    let mut global_stages = BTreeMap::new();

    for entry in &module.entry_points {
        let stage = naga_stages(entry.stage);
        update_stages(module, &entry.function, &mut global_stages, stage);
    }

    global_stages
}

const fn naga_stages(stage: naga::ShaderStage) -> wgpu::ShaderStages {
    match stage {
        naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
        naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
        naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
        naga::ShaderStage::Task => wgpu::ShaderStages::TASK,
        naga::ShaderStage::Mesh => wgpu::ShaderStages::MESH,
    }
}

pub fn entry_stages(module: &naga::Module) -> wgpu::ShaderStages {
    module
        .entry_points
        .iter()
        .map(|entry| naga_stages(entry.stage))
        .collect()
}

fn update_stages_blocks(
    module: &naga::Module,
    block: &naga::Block,
    global_stages: &mut BTreeMap<String, wgpu::ShaderStages>,
    stage: wgpu::ShaderStages,
) {
    for statement in block.iter() {
        match statement {
            naga::Statement::Block(block) => {
                update_stages_blocks(module, block, global_stages, stage);
            }
            naga::Statement::If { accept, reject, .. } => {
                update_stages_blocks(module, accept, global_stages, stage);
                update_stages_blocks(module, reject, global_stages, stage);
            }
            naga::Statement::Switch { cases, .. } => {
                for c in cases {
                    update_stages_blocks(module, &c.body, global_stages, stage);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                update_stages_blocks(module, body, global_stages, stage);
                update_stages_blocks(module, continuing, global_stages, stage);
            }
            naga::Statement::Call { function, .. } => {
                update_stages(module, &module.functions[*function], global_stages, stage);
            }
            _ => (),
        }
    }
}

fn update_stages(
    module: &naga::Module,
    function: &naga::Function,
    global_stages: &mut BTreeMap<String, wgpu::ShaderStages>,
    stage: wgpu::ShaderStages,
) {
    // Search the function body to find function call statements
    update_stages_blocks(module, &function.body, global_stages, stage);

    // Search the function body to find used globals.
    for (_, e) in function.expressions.iter() {
        match e {
            naga::Expression::GlobalVariable(g) => {
                let global = &module.global_variables[*g];
                if let Some(name) = &global.name {
                    let stages = global_stages
                        .entry(name.clone())
                        .or_insert(wgpu::ShaderStages::NONE);
                    *stages = stages.union(stage);
                }
            }
            naga::Expression::CallResult(f) => {
                // Function call expressions
                update_stages(module, &module.functions[*f], global_stages, stage);
            }
            _ => (),
        }
    }
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
        (naga::ScalarKind::Float, 2) => quote!(half::f16),
        (naga::ScalarKind::Float, 4) => quote!(f32),
        (naga::ScalarKind::Float, 8) => quote!(f64),
        // TODO: Do booleans have a width?
        (naga::ScalarKind::Bool, _) => quote!(bool),
        _ => panic!("Unsupported scalar type {scalar:?}"),
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

pub fn rust_type<F>(
    path: &TypePath,
    module: &naga::Module,
    ty: &naga::Type,
    format: MatrixVectorTypes,
    demangle: F,
) -> TokenStream
where
    F: Fn(&str) -> TypePath,
{
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
            let element_type = rust_type(path, module, &module.types[*base], format, demangle);
            let count = Literal::usize_unsuffixed(size.get() as usize);
            quote!([#element_type; #count])
        }
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => {
            panic!(
                "Runtime-sized arrays can only be used in variable declarations or as the last field of a struct."
            );
        }
        naga::TypeInner::Struct {
            members: _,
            span: _,
        } => {
            let member_path = demangle(ty.name.as_ref().unwrap());

            // Use relative paths since we don't know the generated code's root path.
            let name = relative_type_path(path, member_path);

            let name: syn::Path = syn::parse_str(&name).unwrap();
            quote!(#name)
        }
        naga::TypeInner::BindingArray { base: _, size: _ } => todo!(),
        naga::TypeInner::AccelerationStructure { .. } => todo!(),
        naga::TypeInner::RayQuery { .. } => todo!(),
        naga::TypeInner::Array {
            size: naga::ArraySize::Pending(_),
            ..
        } => todo!(),
    }
}

fn relative_type_path(base: &TypePath, target: TypePath) -> String {
    let base_path: PathBuf = base.parent.components.iter().collect();
    let target_path: PathBuf = target.parent.components.iter().collect();
    let relative_path = pathdiff::diff_paths(target_path, base_path).unwrap();

    // TODO: Implement this from scratch with tests?
    let mut components: Vec<_> = relative_path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Prefix(_) => None,
            std::path::Component::RootDir => None,
            std::path::Component::CurDir => None,
            std::path::Component::ParentDir => Some("super".to_string()),
            std::path::Component::Normal(s) => Some(s.to_str()?.to_string()),
        })
        .collect();
    components.push(target.name);

    components.join("::")
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
            (naga::ScalarKind::Float, 2) => wgpu::VertexFormat::Float16,
            (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32,
            (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64,
            _ => unreachable!(),
        },
        naga::TypeInner::Vector { size, scalar } => match size {
            naga::VectorSize::Bi => match (scalar.kind, scalar.width) {
                (naga::ScalarKind::Sint, 1) => wgpu::VertexFormat::Sint8x2,
                (naga::ScalarKind::Uint, 1) => wgpu::VertexFormat::Uint8x2,
                (naga::ScalarKind::Sint, 2) => wgpu::VertexFormat::Sint16x2,
                (naga::ScalarKind::Uint, 2) => wgpu::VertexFormat::Uint16x2,
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x2,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x2,
                (naga::ScalarKind::Float, 2) => wgpu::VertexFormat::Float16x2,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x2,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x2,
                _ => unreachable!(),
            },
            naga::VectorSize::Tri => match (scalar.kind, scalar.width) {
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x3,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x3,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x3,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x3,
                _ => unreachable!(),
            },
            naga::VectorSize::Quad => match (scalar.kind, scalar.width) {
                (naga::ScalarKind::Sint, 1) => wgpu::VertexFormat::Sint8x4,
                (naga::ScalarKind::Uint, 1) => wgpu::VertexFormat::Uint8x4,
                (naga::ScalarKind::Sint, 2) => wgpu::VertexFormat::Sint16x4,
                (naga::ScalarKind::Uint, 2) => wgpu::VertexFormat::Uint16x4,
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x4,
                (naga::ScalarKind::Sint, 4) => wgpu::VertexFormat::Sint32x4,
                (naga::ScalarKind::Float, 2) => wgpu::VertexFormat::Float16x4,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x4,
                (naga::ScalarKind::Float, 8) => wgpu::VertexFormat::Float64x4,
                _ => unimplemented!(),
            },
        },
        _ => todo!(), // are these types even valid as attributes?
    }
}

#[derive(PartialEq, Eq)]
pub struct VertexInput {
    pub name: TypePath,
    pub fields: Vec<(u32, StructMember)>,
}

// TODO: Handle errors.
// Collect the necessary data to generate an equivalent Rust struct.
pub fn get_vertex_input_structs<F>(module: &naga::Module, demangle: F) -> Vec<VertexInput>
where
    F: Fn(&str) -> TypePath + Clone,
{
    let mut structs: Vec<_> = module
        .entry_points
        .iter()
        .filter(|e| e.stage == naga::ShaderStage::Vertex)
        .flat_map(|vertex_entry| vertex_entry_structs(vertex_entry, module, demangle.clone()))
        .collect();

    // Remove structs that are used more than once.
    structs.sort_by_key(|s| s.name.name.clone());
    structs.dedup_by_key(|s| s.name.name.clone());

    structs
}

pub fn vertex_entry_structs<F>(
    vertex_entry: &naga::EntryPoint,
    module: &naga::Module,
    demangle: F,
) -> Vec<VertexInput>
where
    F: Fn(&str) -> TypePath,
{
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
                        name: demangle(arg_type.name.as_ref()?),
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
