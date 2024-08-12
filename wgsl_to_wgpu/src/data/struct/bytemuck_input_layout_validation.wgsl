struct Input0 {
    @size(8)
    a: u32,
    b: i32,
    @align(32)
    c: f32,
    @builtin(vertex_index) d: u32,
};

var<storage, read_write> test: Input0;

struct Outer {
    inner: Inner
}

struct Inner {
    a: f32
}

var<storage, read_write> test2: array<Outer>;

@vertex
fn main(input: Input0) -> vec4<f32> {
    return vec4(0.0);
}