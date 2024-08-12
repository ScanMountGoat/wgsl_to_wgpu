struct Input0 {
    a: u32,
    b: i32,
    c: f32,
};

struct Nested {
    a: Input0,
    b: f32
}

var<workgroup> a: Input0;
var<uniform> b: Nested;

@compute
@workgroup_size(64)
fn main() {}