struct VertexInput0 {};
struct VertexWeight {};
struct Vertices {};
struct VertexWeights {};
struct Transforms {};

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vertex_weights: VertexWeights;
@group(0) @binding(2) var<storage, read_write> dst: Vertices;

@group(1) @binding(0) var<uniform> transforms: Transforms;

@compute
@workgroup_size(64)
fn main() {}