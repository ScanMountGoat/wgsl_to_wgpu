use pretty_assertions::assert_eq;

extern crate wgpu_types as wgpu;

#[test]
fn vertex_entries() {
    // Check vertex entry points and builtin attribute handling.
    let actual = wgsl_to_wgpu::create_shader_module(
        include_str!("wgsl/vertex_entries.wgsl"),
        "shader.wgsl",
        wgsl_to_wgpu::WriteOptions {
            rustfmt: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(actual, include_str!("output/vertex_entries.rs"));
}
