pub const SOURCE: &str = include_str!("bytemuck_input_layout_validation.wgsl");
pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let source = std::borrow::Cow::Borrowed(SOURCE);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source),
    })
}
pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    })
}
pub const ENTRY_MAIN: &str = "main";
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck :: Pod, bytemuck :: Zeroable)]
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
#[derive(Debug, Copy, Clone, PartialEq, bytemuck :: Pod, bytemuck :: Zeroable)]
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
impl Input0 {
    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: std::mem::offset_of!(Input0, a) as u64,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Sint32,
            offset: std::mem::offset_of!(Input0, b) as u64,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32,
            offset: std::mem::offset_of!(Input0, c) as u64,
            shader_location: 2,
        },
    ];
    pub const fn vertex_buffer_layout(
        step_mode: wgpu::VertexStepMode,
    ) -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Input0>() as u64,
            step_mode,
            attributes: &Input0::VERTEX_ATTRIBUTES,
        }
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck :: Pod, bytemuck :: Zeroable)]
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
#[derive(Debug)]
pub struct VertexEntry<const N: usize> {
    pub entry_point: &'static str,
    pub buffers: [wgpu::VertexBufferLayout<'static>; N],
    pub constants: Vec<(&'static str, f64)>,
}
pub fn vertex_state<'a, const N: usize>(
    module: &'a wgpu::ShaderModule,
    entry: &'a VertexEntry<N>,
) -> wgpu::VertexState<'a> {
    wgpu::VertexState {
        module,
        entry_point: Some(entry.entry_point),
        buffers: &entry.buffers,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &entry.constants,
            ..Default::default()
        },
    }
}
pub fn main_entry(input0: wgpu::VertexStepMode) -> VertexEntry<1> {
    VertexEntry {
        entry_point: ENTRY_MAIN,
        buffers: [Input0::vertex_buffer_layout(input0)],
        constants: Default::default(),
    }
}
