#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Input0 {
    pub a: u32,
    pub b: i32,
    pub c: f32,
}
const _: () = assert!(
    std::mem::size_of::<Input0>() == 64,
    "size of Input0 does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Input0, a) == 0,
    "offset of Input0.a does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Input0, b) == 8,
    "offset of Input0.b does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Input0, c) == 32,
    "offset of Input0.c does not match WGSL"
);
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Inner {
    pub a: f32,
}
const _: () = assert!(
    std::mem::size_of::<Inner>() == 4,
    "size of Inner does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Inner, a) == 0,
    "offset of Inner.a does not match WGSL"
);
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Outer {
    pub inner: Inner,
}
const _: () = assert!(
    std::mem::size_of::<Outer>() == 4,
    "size of Outer does not match WGSL"
);
const _: () = assert!(
    std::mem::offset_of!(Outer, inner) == 0,
    "offset of Outer.inner does not match WGSL"
);
