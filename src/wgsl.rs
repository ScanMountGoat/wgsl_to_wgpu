use std::collections::BTreeMap;

pub struct GroupData<'a> {
    pub bindings: Vec<GroupBinding<'a>>,
}

pub struct GroupBinding<'a> {
    pub binding_index: u32,
    pub inner_type: &'a naga::TypeInner,
}

pub fn get_bind_group_data(module: &naga::Module) -> BTreeMap<u32, GroupData> {
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

// TODO: Handle errors.
pub fn get_vertex_input_locations(module: &naga::Module) -> Vec<(String, u32)> {
    let vertex_entry = module
        .entry_points
        .iter()
        .find(|e| e.stage == naga::ShaderStage::Vertex)
        .unwrap();

    let mut shader_locations = Vec::new();

    for argument in &vertex_entry.function.arguments {
        // For entry points, arguments must have a binding unless they are a structure.
        if let Some(binding) = &argument.binding {
            if let naga::Binding::Location { location, .. } = binding {
                shader_locations.push((argument.name.clone().unwrap(), *location));
            }
        } else {
            let arg_type = &module.types[argument.ty];
            match &arg_type.inner {
                naga::TypeInner::Struct {
                    top_level: _,
                    members,
                    span: _,
                } => {
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

    shader_locations
}

#[cfg(test)]
mod test {
    use indoc::indoc;

    use crate::wgsl::get_vertex_input_locations;

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
}
