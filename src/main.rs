wgsl_to_wgpu::wgsl_module!("shader.wgsl");

fn main() {
    let module = naga::front::wgsl::parse_str(include_str!("shader.wgsl")).unwrap();

    // TODO: This should be all the necessary info to generate bind group types.
    for global_handle in module.global_variables.iter() {
        let global = &module.global_variables[global_handle.0];
        if let Some(binding) = &global.binding {
            let inner_type = &module.types[module.global_variables[global_handle.0].ty];
            println!("{:?}", global.name);
            println!("{:?}", inner_type);
            println!("{:?}", binding);
            println!();
        }
    }

    // groups[group].bindings

    for entry_point in module.entry_points.iter() {
        // TODO: Check the binding to determine if the argument is builtin or not.
        // It's probably not necessary to generate types for builtins.

        // TODO: Types can contain other types.

        // The wgsl spec is still a draft, but it appears nested structures aren't allowed for entry points.
        //https://www.w3.org/TR/WGSL/#pipeline-inputs-outputs
        let arg_types: Vec<_> = entry_point
            .function
            .arguments
            .iter()
            .map(|a| &module.types[a.ty])
            .collect();
        // println!("{:?}", entry_point.stage);
        // println!("{:?}", arg_types);
        // println!();
    }
}
