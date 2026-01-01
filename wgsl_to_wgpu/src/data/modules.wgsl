struct uniforms_Uniforms {
    a: vec4<f32>,
    b: uniforms_nested_Nested,
}

struct uniforms_Uniforms2 {
    b: uniforms_nested_Nested,
}

struct uniforms_nested_Nested {
    a: vec4<f32>,
    b: Root,
    c: shared_Shared
}

struct Root {
    c: vec4<f32>
}

struct shared_Shared {
    d: vec4<f32>
}

struct shared_VertexInput {
    @location(0) position: vec3<f32>
}

struct VertexOffset {
    @location(1) offset: vec3<f32>
}

struct shared_VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

const shared_TEST: f32 = 1.0;

@group(0) @binding(0)
var<uniform> bindings_uniforms: uniforms_Uniforms;

@vertex
fn vert(in: shared_VertexInput, offset: VertexOffset) -> shared_VertexOutput {
    var out: shared_VertexOutput;
    out.clip_position = vec4(in.position + offset.offset, shared_TEST);
    return out;
}

@vertex
fn shared_vert(in: shared_VertexInput, offset: VertexOffset) -> shared_VertexOutput {
    var out: shared_VertexOutput;
    out.clip_position = vec4(in.position + offset.offset, shared_TEST);
    return out;
}

@fragment
fn frag(in: shared_VertexOutput) -> @location(0) vec4<f32> {
    return bindings_uniforms.b.a * vec4(0.0);
}