#[repr(C)]
#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    encase::ShaderType,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Input0 {
    pub a: u32,
    pub b: i32,
    pub c: f32,
}
const _: () = assert!(
    std::mem::size_of::<Input0>() == 12,
    "size of Input0 does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Input0, a) == 0,
    "offset of Input0.a does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Input0, b) == 4,
    "offset of Input0.b does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Input0, c) == 8,
    "offset of Input0.c does not match WGSL"
);
#[repr(C)]
#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    bytemuck::Pod,
    bytemuck::Zeroable,
    encase::ShaderType,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Nested {
    pub a: Input0,
    pub b: f32,
}
const _: () = assert!(
    std::mem::size_of::<Nested>() == 16,
    "size of Nested does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Nested, a) == 0,
    "offset of Nested.a does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Nested, b) == 12,
    "offset of Nested.b does not match WGSL"
);
