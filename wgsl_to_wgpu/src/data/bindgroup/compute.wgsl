struct VertexInput0 {
    value: vec4<f32>
};
struct VertexWeight {
    value: vec4<f32>
};
struct Vertices {
    value: vec4<f32>
};
struct VertexWeights {
    value: vec4<f32>
};
struct Transforms {
    value: vec4<f32>
};

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vertex_weights: VertexWeights;
@group(0) @binding(2) var<storage, read_write> dst: Vertices;

@group(1) @binding(0) var<uniform> transforms: Transforms;

@group(2) @binding(0) var<storage, read_write> atomics1: atomic<i32>;
@group(2) @binding(1) var<storage, read_write> atomics2: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> atomics3: array<atomic<u32>, 2>;

@compute
@workgroup_size(64)
fn main() {
    var x = src[0].x;
    x = vertex_weights.value.x;
    x = dst.value.x;
    x = transforms.value.x;
    atomicAdd(&atomics1, 1);
    atomicAdd(&atomics2, 1);
    atomicAdd(&atomics3[1], 1);
}