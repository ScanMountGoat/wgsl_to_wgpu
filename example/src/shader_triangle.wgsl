struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(in.position.xyz * 0.25, 1.0);
    return out;
}

struct shared__Uniforms {
    color_rgb: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> shared__uniforms: shared__Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(shared__uniforms.color_rgb, 1.0);
}
