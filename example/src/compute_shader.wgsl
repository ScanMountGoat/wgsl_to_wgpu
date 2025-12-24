struct Uniforms {
    color_rgb: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> invocation_count: atomic<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // A basic compute shader for demonstrating how to use generated bindings.
    if global_id.x == 0u {
        uniforms.color_rgb = vec3(1.0);
    }
    atomicAdd(&invocation_count, 1);
}
