#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Scalars {
    pub a: u32,
    pub b: i32,
    pub c: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsU32 {
    pub a: [u32; 2],
    pub b: [u32; 3],
    pub c: [u32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsI32 {
    pub a: [i32; 2],
    pub b: [i32; 3],
    pub c: [i32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsF32 {
    pub a: [f32; 2],
    pub b: [f32; 3],
    pub c: [f32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsF64 {
    pub a: [f64; 2],
    pub b: [f64; 3],
    pub c: [f64; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatricesF32 {
    pub a: [[f32; 4]; 4],
    pub b: [[f32; 4]; 3],
    pub c: [[f32; 4]; 2],
    pub d: [[f32; 3]; 4],
    pub e: [[f32; 3]; 3],
    pub f: [[f32; 3]; 2],
    pub g: [[f32; 2]; 4],
    pub h: [[f32; 2]; 3],
    pub i: [[f32; 2]; 2],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatricesF64 {
    pub a: [[f64; 4]; 4],
    pub b: [[f64; 4]; 3],
    pub c: [[f64; 4]; 2],
    pub d: [[f64; 3]; 4],
    pub e: [[f64; 3]; 3],
    pub f: [[f64; 3]; 2],
    pub g: [[f64; 2]; 4],
    pub h: [[f64; 2]; 3],
    pub i: [[f64; 2]; 2],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StaticArrays {
    pub a: [u32; 5],
    pub b: [f32; 3],
    pub c: [[[f32; 4]; 4]; 512],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nested {
    pub a: MatricesF32,
    pub b: MatricesF64,
}
