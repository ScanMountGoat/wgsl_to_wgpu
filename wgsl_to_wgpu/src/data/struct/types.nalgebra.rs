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
    pub a: nalgebra::SVector<u32, 2>,
    pub b: nalgebra::SVector<u32, 3>,
    pub c: nalgebra::SVector<u32, 4>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsI32 {
    pub a: nalgebra::SVector<i32, 2>,
    pub b: nalgebra::SVector<i32, 3>,
    pub c: nalgebra::SVector<i32, 4>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsF32 {
    pub a: nalgebra::SVector<f32, 2>,
    pub b: nalgebra::SVector<f32, 3>,
    pub c: nalgebra::SVector<f32, 4>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VectorsF64 {
    pub a: nalgebra::SVector<f64, 2>,
    pub b: nalgebra::SVector<f64, 3>,
    pub c: nalgebra::SVector<f64, 4>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatricesF32 {
    pub a: nalgebra::SMatrix<f32, 4, 4>,
    pub b: nalgebra::SMatrix<f32, 3, 4>,
    pub c: nalgebra::SMatrix<f32, 2, 4>,
    pub d: nalgebra::SMatrix<f32, 4, 3>,
    pub e: nalgebra::SMatrix<f32, 3, 3>,
    pub f: nalgebra::SMatrix<f32, 2, 3>,
    pub g: nalgebra::SMatrix<f32, 4, 2>,
    pub h: nalgebra::SMatrix<f32, 3, 2>,
    pub i: nalgebra::SMatrix<f32, 2, 2>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatricesF64 {
    pub a: nalgebra::SMatrix<f64, 4, 4>,
    pub b: nalgebra::SMatrix<f64, 3, 4>,
    pub c: nalgebra::SMatrix<f64, 2, 4>,
    pub d: nalgebra::SMatrix<f64, 4, 3>,
    pub e: nalgebra::SMatrix<f64, 3, 3>,
    pub f: nalgebra::SMatrix<f64, 2, 3>,
    pub g: nalgebra::SMatrix<f64, 4, 2>,
    pub h: nalgebra::SMatrix<f64, 3, 2>,
    pub i: nalgebra::SMatrix<f64, 2, 2>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StaticArrays {
    pub a: [u32; 5],
    pub b: [f32; 3],
    pub c: [nalgebra::SMatrix<f32, 4, 4>; 512],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nested {
    pub a: MatricesF32,
    pub b: MatricesF64,
}
