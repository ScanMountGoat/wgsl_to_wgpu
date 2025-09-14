use super::*;
use indoc::indoc;

// Tokenstreams can't be compared directly using PartialEq.
// Use pretty_print to normalize the formatting and compare strings.
// Use a colored diff output to make differences easier to see.
#[cfg(test)]
#[macro_export]
macro_rules! assert_tokens_eq {
    ($a:expr, $b:expr) => {
        pretty_assertions::assert_eq!(
            $crate::pretty_print_rustfmt($a),
            $crate::pretty_print_rustfmt($b)
        )
    };
}

#[cfg(test)]
#[macro_export]
macro_rules! assert_tokens_snapshot {
    ($output:expr) => {
        let mut settings = insta::Settings::new();
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        settings.bind(|| {
            insta::assert_snapshot!($crate::pretty_print_rustfmt($output));
        });
    };
}

#[cfg(test)]
#[macro_export]
macro_rules! assert_rust_snapshot {
    ($output:expr) => {
        let mut settings = insta::Settings::new();
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        settings.bind(|| {
            insta::assert_snapshot!($output);
        });
    };
}

#[test]
fn create_shader_module_push_constants() {
    let source = indoc! {r#"
        var<push_constant> consts: vec4<f32>;

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return consts;
        }
    "#};

    let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
    assert_rust_snapshot!(actual);
}

#[test]
fn create_shader_multiple_entries() {
    let source = indoc! {r#"
        @group(0) @binding(0) var<uniform> a: f32;
        @group(0) @binding(1) var<uniform> b: f32;
        @group(0) @binding(2) var<uniform> c: f32;
        @group(0) @binding(3) var<uniform> d: u32;
        @group(0) @binding(4) var<uniform> e: u32;
        @group(0) @binding(5) var<uniform> f: u32;
        @group(0) @binding(6) var<uniform> g: u32;
        @group(0) @binding(7) var<uniform> h: u32;
        @group(0) @binding(8) var<uniform> i: f64;

        fn inner() -> f32 {
            return d;
        }

        fn inner_double() -> f64 {
            return i;
        }

        @vertex
        fn vs_main() {
            {
                let x = b;
                let y = f;
            }

            let x = h;

            switch e {
                default: {
                    let y = e;
                    return;
                }
            }
        }

        @fragment
        fn fs_main()  {
            let z = e;

            loop {
                let z = c;
            }

            if true {
                let x = h;
                let y = g;
            }
        }

        @compute @workgroup_size(1, 1, 1)
        fn main() {
            let y = inner();
            let z = f;
            loop {
                let w = g;
                let x = h;
            }
        }
    "#};

    let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
    assert_rust_snapshot!(actual);
}

#[test]
fn create_shader_module_multiple_outputs() {
    let source = indoc! {r#"
        struct Output {
            @location(0) col0: vec4<f32>,
            @builtin(frag_depth) depth: f32,
            @location(1) col1: vec4<f32>,
        };

        @fragment
        fn fs_multiple() -> Output {}
    "#};
    let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
    assert_rust_snapshot!(actual);
}

#[test]
fn create_shader_modules_source() {
    let source = "@fragment fn main() {}";
    let actual =
        create_shader_modules(source, WriteOptions::default(), demangle_identity).unwrap();
    assert_rust_snapshot!(actual);
}

#[test]
fn create_shader_modules_source_rustfmt() {
    let source = "@fragment fn main() {}";
    let actual = create_shader_modules(
        source,
        WriteOptions {
            rustfmt: true,
            ..Default::default()
        },
        demangle_identity,
    )
    .unwrap();
    assert_rust_snapshot!(actual);
}

#[test]
fn create_shader_module_consecutive_bind_groups() {
    let source = indoc! {r#"
        struct A {
            f: vec4<f32>
        };
        @group(0) @binding(0) var<uniform> a: A;
        @group(1) @binding(0) var<uniform> b: f32;
        @group(2) @binding(0) var<uniform> c: vec4<f32>;
        @group(3) @binding(0) var<uniform> d: mat4x4<f32>;

        @vertex
        fn vs_main() {}

        @fragment
        fn fs_main() {}
    "#};

    create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
}

#[test]
fn create_shader_module_non_consecutive_bind_groups() {
    let source = indoc! {r#"
        @group(0) @binding(0) var<uniform> a: vec4<f32>;
        @group(1) @binding(0) var<uniform> b: vec4<f32>;
        @group(3) @binding(0) var<uniform> c: vec4<f32>;

        @fragment
        fn main() {}
    "#};

    let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());
    assert!(matches!(
        result,
        Err(CreateModuleError::NonConsecutiveBindGroups)
    ));
}

#[test]
fn create_shader_module_repeated_bindings() {
    let source = indoc! {r#"
        struct A {
            f: vec4<f32>
        };
        @group(0) @binding(2) var<uniform> a: A;
        @group(0) @binding(2) var<uniform> b: A;

        @fragment
        fn main() {}
    "#};

    let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());
    assert!(matches!(
        result,
        Err(CreateModuleError::DuplicateBinding { binding: 2 })
    ));
}

fn items_to_tokens(items: Vec<(TypePath, TokenStream)>) -> TokenStream {
    let mut root = Module::default();
    root.add_module_items(&items, &ModulePath::default());
    root.to_tokens()
}

#[test]
fn write_vertex_module_empty() {
    let source = indoc! {r#"
        @vertex
        fn main() {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = vertex_struct_methods(&module, demangle_identity);

    assert_tokens_eq!(quote!(), items_to_tokens(actual));
}

#[test]
fn write_vertex_module_single_input_float32() {
    let source = indoc! {r#"
        struct VertexInput0 {
            @location(0) a: f32,
            @location(1) b: vec2<f32>,
            @location(2) c: vec3<f32>,
            @location(3) d: vec4<f32>,
        };

        @vertex
        fn main(in0: VertexInput0) {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = vertex_struct_methods(&module, demangle_identity);

    assert_tokens_snapshot!(items_to_tokens(actual));
}

#[test]
fn write_vertex_module_single_input_float64() {
    let source = indoc! {r#"
        struct VertexInput0 {
            @location(0) a: f64,
            @location(1) b: vec2<f64>,
            @location(2) c: vec3<f64>,
            @location(3) d: vec4<f64>,
        };

        @vertex
        fn main(in0: VertexInput0) {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = vertex_struct_methods(&module, demangle_identity);

    assert_tokens_snapshot!(items_to_tokens(actual));
}

#[test]
fn write_vertex_module_single_input_float16() {
    let source = indoc! {r#"
        enable f16;

        struct VertexInput0 {
            @location(0) a: f16,
            @location(1) b: vec2<f16>,
            @location(2) c: vec4<f16>,
        };

        @vertex
        fn main(in0: VertexInput0) {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = vertex_struct_methods(&module, demangle_identity);

    assert_tokens_snapshot!(items_to_tokens(actual));
}

#[test]
fn write_vertex_module_single_input_sint32() {
    let source = indoc! {r#"
        struct VertexInput0 {
            @location(0) a: i32,
            @location(1) b: vec2<i32>,
            @location(2) c: vec3<i32>,
            @location(3) d: vec4<i32>,

        };

        @vertex
        fn main(in0: VertexInput0) {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = vertex_struct_methods(&module, demangle_identity);

    assert_tokens_snapshot!(items_to_tokens(actual));
}

#[test]
fn write_vertex_module_single_input_uint32() {
    let source = indoc! {r#"
        struct VertexInput0 {
            @location(0) a: u32,
            @location(1) b: vec2<u32>,
            @location(2) c: vec3<u32>,
            @location(3) d: vec4<u32>,
        };

        @vertex
        fn main(in0: VertexInput0) {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = vertex_struct_methods(&module, demangle_identity);

    assert_tokens_snapshot!(items_to_tokens(actual));
}

#[test]
fn write_compute_module_empty() {
    let source = indoc! {r#"
        @vertex
        fn main() {}
    "#};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = compute_module(&module, demangle_identity);

    assert_tokens_eq!(quote!(), actual);
}

#[test]
fn write_compute_module_multiple_entries() {
    let source = indoc! {r#"
        @compute
        @workgroup_size(1,2,3)
        fn main1() {}

        @compute
        @workgroup_size(256)
        fn main2() {}
    "#
    };

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let actual = compute_module(&module, demangle_identity);

    assert_tokens_snapshot!(actual);
}

#[test]
fn create_shader_module_parse_error() {
    let source = indoc! {r#"
        var<push_constant> consts: vec4<f32>;

        @fragment
        fn fs_main() }
    "#};

    let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());

    assert!(matches!(result, Err(CreateModuleError::ParseError { .. })));
}

#[test]
fn create_shader_module_semantic_error() {
    let source = indoc! {r#"
        var<push_constant> consts: vec4<f32>;

        @fragment
        fn fs_main() {
            consts.x = 1;
        }
    "#};

    let result = create_shader_module(
        source,
        "shader.wgsl",
        WriteOptions {
            validate: Some(Default::default()),
            ..Default::default()
        },
    );

    assert!(matches!(
        result,
        Err(CreateModuleError::ValidationError { .. })
    ));
}

fn demangle_underscore(name: &str) -> TypePath {
    // Preprocessors that support modules mangle absolute paths.
    // Use a very basic mangling scheme that assumes no '_' in the identifier name.
    // This allows testing the module logic without needing extra dependencies.
    // a_b_C -> a::b::C
    let components: Vec<_> = name.split("_").collect();
    let (name, parents) = components.split_last().unwrap();
    TypePath {
        parent: ModulePath {
            components: parents.iter().map(|p| p.to_string()).collect(),
        },
        name: name.to_string(),
    }
}

#[test]
fn single_root_module() {
    let output = create_shader_modules(
        include_str!("data/modules.wgsl"),
        WriteOptions {
            rustfmt: true,
            ..Default::default()
        },
        demangle_underscore,
    )
    .unwrap();

    assert_rust_snapshot!(output);
}

#[test]
fn add_single_root_module() {
    let mut root = Module::default();
    let options = WriteOptions {
        rustfmt: true,
        ..Default::default()
    };
    root.add_shader_module(
        include_str!("data/modules.wgsl"),
        None,
        options,
        ModulePath::default(),
        demangle_underscore,
    )
    .unwrap();

    let output = root.to_generated_bindings(options);
    assert_rust_snapshot!(output);
}

#[test]
fn add_duplicate_module_different_paths() {
    // Test shared types and handling of duplicate names.
    let mut root = Module::default();
    let options = WriteOptions {
        rustfmt: true,
        ..Default::default()
    };
    root.add_shader_module(
        include_str!("data/modules.wgsl"),
        None,
        options,
        ModulePath {
            components: vec!["shader1".to_string()],
        },
        demangle_underscore,
    )
    .unwrap();
    root.add_shader_module(
        include_str!("data/modules.wgsl"),
        None,
        options,
        ModulePath {
            components: vec!["shaders".to_string(), "shader2".to_string()],
        },
        demangle_underscore,
    )
    .unwrap();

    let output = root.to_generated_bindings(options);
    assert_rust_snapshot!(output);
}

#[test]
fn vertex_entries() {
    // Check vertex entry points and builtin attribute handling.
    let actual = create_shader_module(
        include_str!("data/vertex_entries.wgsl"),
        "shader.wgsl",
        WriteOptions {
            rustfmt: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_rust_snapshot!(actual);
}

#[test]
fn shader_stage_collection() {
    // Check the visibility: wgpu::ShaderStages::COMPUTE
    let actual = create_shader_module(
        include_str!("data/shader_stage_collection.wgsl"),
        "shader.wgsl",
        WriteOptions {
            rustfmt: true,
            derive_encase_host_shareable: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_rust_snapshot!(actual);
}
