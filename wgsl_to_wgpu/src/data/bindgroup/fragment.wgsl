struct Transforms {
    value: vec4<f32>
};

@group(0) @binding(0) var<uniform> transforms: Transforms;

@fragment
fn fs_main() {
    let x = transforms.value.x;
}