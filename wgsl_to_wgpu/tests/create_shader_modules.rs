use pretty_assertions::assert_eq;
use wgsl_to_wgpu::{Module, ModulePath, TypePath};

fn demangle_underscore(name: &str) -> TypePath {
    // Preprocessors that support modules mangle absolute paths.
    // Use a very basic mangling scheme that assumes no '_' in the identifier name.
    // This allows testing the module logic without needing extra dependencies.
    // a_b_C -> a::b::C
    let components: Vec<_> = name.split("_").collect();
    let (name, parents) = components.split_last().unwrap();
    TypePath {
        parent: ModulePath {
            components: parents.into_iter().map(|p| p.to_string()).collect(),
        },
        name: name.to_string(),
    }
}

#[test]
fn single_root_module() {
    let output = wgsl_to_wgpu::create_shader_modules(
        include_str!("wgsl/modules.wgsl"),
        wgsl_to_wgpu::WriteOptions {
            rustfmt: true,
            ..Default::default()
        },
        demangle_underscore,
    )
    .unwrap();
    assert_eq!(include_str!("output/modules_single.rs"), output);
}

#[test]
fn add_single_root_module() {
    let mut root = Module::default();
    let options = wgsl_to_wgpu::WriteOptions {
        rustfmt: true,
        ..Default::default()
    };
    root.add_shader_module(
        include_str!("wgsl/modules.wgsl"),
        None,
        options,
        ModulePath::default(),
        demangle_underscore,
    )
    .unwrap();

    let output = root.to_generated_bindings(options);
    assert_eq!(include_str!("output/modules_single.rs"), output);
}

#[test]
fn add_duplicate_module_different_paths() {
    // Test shared types and handling of duplicate names.
    let mut root = Module::default();
    let options = wgsl_to_wgpu::WriteOptions {
        rustfmt: true,
        ..Default::default()
    };
    root.add_shader_module(
        include_str!("wgsl/modules.wgsl"),
        None,
        options,
        ModulePath {
            components: vec!["shader1".to_string()],
        },
        demangle_underscore,
    )
    .unwrap();
    root.add_shader_module(
        include_str!("wgsl/modules.wgsl"),
        None,
        options,
        ModulePath {
            components: vec!["shaders".to_string(), "shader2".to_string()],
        },
        demangle_underscore,
    )
    .unwrap();

    let output = root.to_generated_bindings(options);
    assert_eq!(include_str!("output/modules.rs"), output);
}
