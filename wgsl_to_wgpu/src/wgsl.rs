use naga::StructMember;
use std::collections::{BTreeMap};

use crate::CreateModuleError;

pub struct GroupData<'a> {
    pub bindings: Vec<GroupBinding<'a>>,
}

pub struct GroupBinding<'a> {
    pub name: Option<String>,
    pub binding_index: u32,
    pub binding_type: &'a naga::Type,
    pub storage_class: naga::StorageClass,
}

// TODO: Improve error handling/error reporting.
pub fn shader_stages(module: &naga::Module) -> wgpu::ShaderStages {
    let mut shader_stages = wgpu::ShaderStages::NONE;
    for entry in &module.entry_points {
        match entry.stage {
            naga::ShaderStage::Vertex => shader_stages.insert(wgpu::ShaderStages::VERTEX),
            naga::ShaderStage::Fragment => shader_stages.insert(wgpu::ShaderStages::FRAGMENT),
            naga::ShaderStage::Compute => shader_stages.insert(wgpu::ShaderStages::COMPUTE),
        }
    }
    shader_stages
}

fn rust_scalar_type(kind: naga::ScalarKind, width: u8) -> String {
    // TODO: Support other widths?
    match (kind, width) {
        (naga::ScalarKind::Sint, 4) => "i32".to_string(),
        (naga::ScalarKind::Uint, 4) => "u32".to_string(),
        (naga::ScalarKind::Float, 4) => "f32".to_string(),
        // TODO: Do booleans have a width?
        (naga::ScalarKind::Bool, _) => "bool".to_string(),
        _ => todo!(),
    }
}

pub fn buffer_binding_type(storage: naga::StorageClass) -> String {
    match storage {
        naga::StorageClass::Uniform => "wgpu::BufferBindingType::Uniform".to_string(),
        naga::StorageClass::Storage { access } => {
            let _is_read = access.contains(naga::StorageAccess::LOAD);
            let is_write = access.contains(naga::StorageAccess::STORE);

            // TODO: Is this correct?
            if is_write {
                "wgpu::BufferBindingType::Storage { read_only: false }".to_string()
            } else {
                "wgpu::BufferBindingType::Storage { read_only: true }".to_string()
            }
        }
        _ => todo!(),
    }
}

pub fn rust_type(module: &naga::Module, ty: &naga::Type) -> String {
    match &ty.inner {
        naga::TypeInner::Scalar { kind, width } => rust_scalar_type(*kind, *width),
        naga::TypeInner::Vector { size, kind, width } => match size {
            naga::VectorSize::Bi => format!("[{}; 2]", rust_scalar_type(*kind, *width)),
            naga::VectorSize::Tri => format!("[{}; 3]", rust_scalar_type(*kind, *width)),
            naga::VectorSize::Quad => format!("[{}; 4]", rust_scalar_type(*kind, *width)),
        },
        naga::TypeInner::Matrix {
            columns,
            rows,
            width,
        } => match (rows, columns, width) {
            // TODO: Don't force glam here?
            (naga::VectorSize::Quad, naga::VectorSize::Quad, 4) => "glam::Mat4".to_string(),
            _ => todo!(),
        },
        naga::TypeInner::Image { .. } => todo!(),
        naga::TypeInner::Sampler { .. } => todo!(),
        naga::TypeInner::Atomic { kind: _, width: _ } => todo!(),
        naga::TypeInner::Pointer { base: _, class: _ } => todo!(),
        naga::TypeInner::ValuePointer {
            size: _,
            kind: _,
            width: _,
            class: _,
        } => todo!(),
        naga::TypeInner::Array { base, size, stride: _ } => {
            // TODO: Support arrays other than arrays with a static size?
            let element_type = rust_type(module, &module.types[*base]);
            let count = array_length(size, module);
            format!("[{element_type}; {count}]")
        }
        naga::TypeInner::Struct { members: _, span: _ } => {
            // TODO: Support structs?
            ty.name.as_ref().unwrap().to_string()
        }
    }
}

pub fn vertex_format(ty: &naga::Type) -> wgpu::VertexFormat {
    // Not all wgsl types work as vertex attributes in wgpu.
    match &ty.inner {
        naga::TypeInner::Scalar { kind: _, width: _ } => todo!(),
        naga::TypeInner::Vector { size, kind, width } => match size {
            naga::VectorSize::Bi => match (kind, width) {
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x2,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x2,
                _ => todo!(),
            },
            naga::VectorSize::Tri => match (kind, width) {
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x3,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x3,
                _ => todo!(),
            },
            naga::VectorSize::Quad => match (kind, width) {
                (naga::ScalarKind::Uint, 4) => wgpu::VertexFormat::Uint32x4,
                (naga::ScalarKind::Float, 4) => wgpu::VertexFormat::Float32x4,
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

pub fn get_bind_group_data(module: &naga::Module) -> Result<BTreeMap<u32, GroupData>, CreateModuleError> {
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
                storage_class: global.class,
            };
            // Repeated bindings will probably cause a compile error.
            // We'll still check for it here just in case.
            if group.bindings.iter().any(|g| g.binding_index == binding.binding) {
                return Err(CreateModuleError::DuplicateBinding { binding: binding.binding });
            }
            group.bindings.push(group_binding);
        }
    }

    // wgpu expects bind groups to be consecutive starting from 0.
    // TODO: Use a result instead?
    if groups.iter().map(|(i, _)| *i as usize).eq(0..groups.len()) {
        Ok(groups)
    } else {
        Err(CreateModuleError::NonConsecutiveBindGroups)
    }
}

pub struct VertexInput {
    pub name: String,
    pub fields: Vec<(u32, StructMember)>,
}

// TODO: Handle errors.
// Collect the necessary data to generate an equivalent Rust struct.
pub fn get_vertex_input_structs(module: &naga::Module) -> Vec<VertexInput> {
    let mut structs = Vec::new();

    // TODO: Just map/collect?
    if let Some(vertex_entry) = module
        .entry_points
        .iter()
        .find(|e| e.stage == naga::ShaderStage::Vertex)
    {
        for argument in &vertex_entry.function.arguments {
            // For entry points, arguments must have a binding unless they are a structure.
            if let Some(_binding) = &argument.binding {
                // TODO: How to create a structure for regular bindings?
            } else {
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

                        structs.push(input);
                    }
                    // This case should be prevented by the checks above.
                    _ => unreachable!(),
                }
            }
        }
    }

    structs
}

pub fn get_vertex_input_locations(module: &naga::Module) -> Vec<(String, u32)> {
    let mut shader_locations = Vec::new();

    // TODO: Just map/collect?
    if let Some(vertex_entry) = module
        .entry_points
        .iter()
        .find(|e| e.stage == naga::ShaderStage::Vertex)
    {
        for argument in &vertex_entry.function.arguments {
            // For entry points, arguments must have a binding unless they are a structure.
            if let Some(binding) = &argument.binding {
                if let naga::Binding::Location { location, .. } = binding {
                    shader_locations.push((argument.name.clone().unwrap(), *location));
                }
            } else {
                let arg_type = &module.types[argument.ty];
                match &arg_type.inner {
                    naga::TypeInner::Struct { members, span: _ } => {
                        for member in members {
                            match member.binding.as_ref().unwrap() {
                                naga::Binding::BuiltIn(_) => (),
                                naga::Binding::Location { location, .. } => {
                                    shader_locations.push((member.name.clone().unwrap(), *location))
                                }
                            }
                        }
                    }
                    // This case should be prevented by the checks above.
                    _ => unreachable!(),
                }
            }
        }
    }

    shader_locations
}

#[cfg(test)]
mod test {
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
            [[stage(vertex)]]
            fn main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::VERTEX, shader_stages(&module));
    }

    #[test]
    fn shader_stages_fragment() {
        let source = indoc! {r#"
            [[stage(fragment)]]
            fn main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::FRAGMENT, shader_stages(&module));
    }

    #[test]
    fn shader_stages_vertex_fragment() {
        let source = indoc! {r#"
            [[stage(vertex)]]
            fn vs_main()  {}

            [[stage(fragment)]]
            fn fs_main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::VERTEX_FRAGMENT, shader_stages(&module));
    }

    #[test]
    fn shader_stages_compute() {
        let source = indoc! {r#"
            [[stage(compute)]]
            fn main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::COMPUTE, shader_stages(&module));
    }

    #[test]
    fn shader_stages_all() {
        let source = indoc! {r#"
            [[stage(vertex)]]
            fn vs_main()  {}

            [[stage(fragment)]]
            fn fs_main()  {}

            [[stage(compute)]]
            fn cs_main()  {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(wgpu::ShaderStages::all(), shader_stages(&module));
    }

    #[test]
    fn vertex_input_structs_two_structs() {
        let source = indoc! {r#"
            struct VertexInput0 {
                [[location(0)]] in0: vec4<f32>;
                [[location(1)]] in1: vec4<f32>;
                [[location(2)]] in2: vec4<f32>;
            };
            
            struct VertexInput1 {
                [[location(3)]] in3: vec4<f32>;
                [[location(4)]] in4: vec4<f32>;
                [[location(5)]] in5: vec4<f32>;
                [[location(6)]] in6: vec4<u32>;
            };

            [[stage(vertex)]]
            fn main(
                in0: VertexInput0,
                in1: VertexInput1
            ) -> [[builtin(position)]] vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let vertex_inputs = get_vertex_input_structs(&module);
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

    #[test]
    fn vertex_locations_struct_two_fields() {
        let source = indoc! {r#"
            struct VertexInput {
                [[location(0)]] position: vec3<f32>;
                [[location(1)]] tex_coords: vec2<f32>;
            };

            [[stage(vertex)]]
            fn main(
                model: VertexInput,
            ) -> [[builtin(position)]] vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let shader_locations = get_vertex_input_locations(&module);
        assert_eq!(
            &[("position".to_string(), 0), ("tex_coords".to_string(), 1)],
            &shader_locations[..]
        );
    }

    #[test]
    fn vertex_locations_struct_no_fields() {
        let source = indoc! {r#"
            struct VertexInput {
            };

            [[stage(vertex)]]
            fn main(
                model: VertexInput,
            ) -> [[builtin(position)]] vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let shader_locations = get_vertex_input_locations(&module);
        assert!(shader_locations.is_empty());
    }

    #[test]
    fn vertex_locations_struct_builtin_field() {
        let source = indoc! {r#"
            struct VertexInput {
                [[builtin(vertex_index)]] VertexIndex : u32;
            };

            [[stage(vertex)]]
            fn main(
                model: VertexInput,
            ) -> [[builtin(position)]] vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let shader_locations = get_vertex_input_locations(&module);
        assert!(shader_locations.is_empty());
    }

    #[test]
    fn vertex_locations_struct_builtin_parameter() {
        let source = indoc! {r#"
            [[stage(vertex)]]
            fn main(
                [[builtin(vertex_index)]] VertexIndex : u32,
            ) -> [[builtin(position)]] vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let shader_locations = get_vertex_input_locations(&module);
        assert!(shader_locations.is_empty());
    }

    #[test]
    fn vertex_locations_two_parameters() {
        let source = indoc! {r#"
            [[stage(vertex)]]
            fn main([[location(0)]] position: vec4<f32>,
                    [[location(1)]] tex_coords: vec2<f32>
            ) -> [[builtin(position)]] vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#};
        let module = naga::front::wgsl::parse_str(source).unwrap();

        let shader_locations = get_vertex_input_locations(&module);
        assert_eq!(
            &[("position".to_string(), 0), ("tex_coords".to_string(), 1)],
            &shader_locations[..]
        );
    }

    #[test]
    fn bind_group_data_consecutive_bind_groups() {
        let source = indoc! {r#"
            [[group(0), binding(0)]] var<uniform> a: vec4<f32>;
            [[group(1), binding(0)]] var<uniform> b: vec4<f32>;
            [[group(2), binding(0)]] var<uniform> c: vec4<f32>;

            [[stage(fragment)]]
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert_eq!(3, get_bind_group_data(&module).unwrap().len());
    }

    #[test]
    fn bind_group_data_first_group_not_zero() {
        let source = indoc! {r#"
            [[group(1), binding(0)]] var<uniform> a: vec4<f32>;

            [[stage(fragment)]]
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert!(matches!(get_bind_group_data(&module), Err(CreateModuleError::NonConsecutiveBindGroups)));
    }

    #[test]
    fn bind_group_data_non_consecutive_bind_groups() {
        let source = indoc! {r#"
            [[group(0), binding(0)]] var<uniform> a: vec4<f32>;
            [[group(1), binding(0)]] var<uniform> b: vec4<f32>;
            [[group(3), binding(0)]] var<uniform> c: vec4<f32>;

            [[stage(fragment)]]
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert!(matches!(get_bind_group_data(&module), Err(CreateModuleError::NonConsecutiveBindGroups)));
    }
}
