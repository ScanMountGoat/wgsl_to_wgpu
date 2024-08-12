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
    pub a: glam::UVec2,
    pub b: glam::UVec3,
    pub c: glam::UVec4,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsI32 {
    pub a: glam::IVec2,
    pub b: glam::IVec3,
    pub c: glam::IVec4,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsF32 {
    pub a: glam::Vec2,
    pub b: glam::Vec3,
    pub c: glam::Vec4,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsF64 {
    pub a: glam::DVec2,
    pub b: glam::DVec3,
    pub c: glam::DVec4,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatricesF32 {
    pub a: glam::Mat4,
    pub b: [[f32; 4]; 3],
    pub c: [[f32; 4]; 2],
    pub d: [[f32; 3]; 4],
    pub e: glam::Mat3,
    pub f: [[f32; 3]; 2],
    pub g: [[f32; 2]; 4],
    pub h: [[f32; 2]; 3],
    pub i: glam::Mat2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatricesF64 {
    pub a: glam::DMat4,
    pub b: [[f64; 4]; 3],
    pub c: [[f64; 4]; 2],
    pub d: [[f64; 3]; 4],
    pub e: glam::DMat3,
    pub f: [[f64; 3]; 2],
    pub g: [[f64; 2]; 4],
    pub h: [[f64; 2]; 3],
    pub i: glam::DMat2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StaticArrays {
    pub a: [u32; 5],
    pub b: [f32; 3],
    pub c: [glam::Mat4; 512],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nested {
    pub a: MatricesF32,
    pub b: MatricesF64,
}
