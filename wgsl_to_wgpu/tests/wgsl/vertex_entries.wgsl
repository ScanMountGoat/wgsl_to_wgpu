struct Input0 {
    @location(0) in0: vec4<f32>,
    @location(1) in1: vec4<f32>,
    @location(2) in2: vec4<f32>,
};

struct Input1 {
    @location(3) in3: vec4<f32>,
    @location(4) in4: vec4<f32>,
    @builtin(vertex_index) index: u32,
    @location(5) in5: vec4<f32>,
    @location(6) in6: vec4<u32>,
};

@vertex
fn vs_main_none() -> @builtin(position) vec4<f32> {
    return vec4(0.0);
}

@vertex
fn vs_main_single(in0: Input0) -> @builtin(position) vec4<f32> {
    return vec4(0.0);
}

@vertex
fn vs_main_multiple(
    in0: Input0,
    in1: Input1,
    @builtin(instance_index) in2: u32,
    @location(7) in3: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}
