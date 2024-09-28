struct Transforms {
    value: vec4<f32>
};

@group(0) @binding(0) var<uniform> transforms: Transforms;

@vertex
fn vs_main() {
    let x = transforms.value.x;
}