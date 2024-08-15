struct Input0 {
    a: u32,
    b: i32,
    c: f32,
};

struct Nested {
    a: Input0,
    b: f32
}

var<uniform> a: Input0;
var<storage, read> b: Nested;

@fragment
fn main() {}