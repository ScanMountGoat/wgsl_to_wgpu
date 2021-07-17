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
// TODO: Create a separate module for dealing with naga types.
// TODO: It should be straightforward to add tests for this by harcoding small shader modules.
pub fn get_vertex_input_locations(module: &naga::Module) -> Vec<(String, u32)> {
    let vertex_entry = module
        .entry_points
        .iter()
        .find(|e| e.stage == naga::ShaderStage::Vertex)
        .unwrap();

    let mut shader_locations = Vec::new();
    // TODO: Support non structs by also checking the arguments.
    // Arguments must have a binding unless they are a structure.
    let arg_types: Vec<_> = vertex_entry
        .function
        .arguments
        .iter()
        .map(|a| &module.types[a.ty])
        .collect();
    for arg_type in arg_types {
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
            _ => (),
        }
    }

    shader_locations
}
