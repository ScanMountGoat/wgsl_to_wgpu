enable f16;

struct Scalars {
    a: u32,
    b: i32,
    c: f32,
    d: f16
};
var<uniform> a: Scalars;

struct VectorsU32 {
    a: vec2<u32>,
    b: vec3<u32>,
    c: vec4<u32>,
};
var<uniform> b: VectorsU32;

struct VectorsI32 {
    a: vec2<i32>,
    b: vec3<i32>,
    c: vec4<i32>,
};
var<uniform> c: VectorsI32;

struct VectorsF32 {
    a: vec2<f32>,
    b: vec3<f32>,
    c: vec4<f32>,
};
var<uniform> d: VectorsF32;

struct VectorsF64 {
    a: vec2<f64>,
    b: vec3<f64>,
    c: vec4<f64>,
};
var<uniform> e: VectorsF64;

struct MatricesF32 {
    a: mat4x4<f32>,
    b: mat4x3<f32>,
    c: mat4x2<f32>,
    d: mat3x4<f32>,
    e: mat3x3<f32>,
    f: mat3x2<f32>,
    g: mat2x4<f32>,
    h: mat2x3<f32>,
    i: mat2x2<f32>,
};
var<uniform> f: MatricesF32;

struct MatricesF64 {
    a: mat4x4<f64>,
    b: mat4x3<f64>,
    c: mat4x2<f64>,
    d: mat3x4<f64>,
    e: mat3x3<f64>,
    f: mat3x2<f64>,
    g: mat2x4<f64>,
    h: mat2x3<f64>,
    i: mat2x2<f64>,
};
var<uniform> g: MatricesF64;

struct StaticArrays {
    a: array<u32, 5>,
    b: array<f32, 3>,
    c: array<mat4x4<f32>, 512>,
};
var<uniform> h: StaticArrays;

struct Nested {
    a: MatricesF32,
    b: MatricesF64
}
var<uniform> i: Nested;

struct VectorsF16 {
    a: vec2<f16>,
    b: vec4<f16>,
};
var<uniform> j: VectorsF16;

struct MatricesF16 {
    a: mat4x4<f16>,
    b: mat4x3<f16>,
    c: mat4x2<f16>,
    d: mat3x4<f16>,
    e: mat3x3<f16>,
    f: mat3x2<f16>,
    g: mat2x4<f16>,
    h: mat2x3<f16>,
    i: mat2x2<f16>,
};
var<uniform> k: MatricesF16;

struct Atomics {
    num: atomic<u32>,
    numi: atomic<i32>,
};
var <storage, read_write> atomics: Atomics;

@fragment
fn main() {}