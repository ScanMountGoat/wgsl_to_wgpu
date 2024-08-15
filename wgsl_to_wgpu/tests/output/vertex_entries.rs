#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Input0 {
    pub in0: [f32; 4],
    pub in1: [f32; 4],
    pub in2: [f32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Input1 {
    pub in3: [f32; 4],
    pub in4: [f32; 4],
    pub in5: [f32; 4],
    pub in6: [u32; 4],
}
pub const ENTRY_VS_MAIN_NONE: &str = "vs_main_none";
pub const ENTRY_VS_MAIN_SINGLE: &str = "vs_main_single";
pub const ENTRY_VS_MAIN_MULTIPLE: &str = "vs_main_multiple";
#[derive(Debug)]
pub struct VertexEntry<const N: usize> {
    pub entry_point: &'static str,
    pub buffers: [wgpu::VertexBufferLayout<'static>; N],
    pub constants: std::collections::HashMap<String, f64>,
}
pub fn vertex_state<'a, const N: usize>(
    module: &'a wgpu::ShaderModule,
    entry: &'a VertexEntry<N>,
) -> wgpu::VertexState<'a> {
    wgpu::VertexState {
        module,
        entry_point: entry.entry_point,
        buffers: &entry.buffers,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &entry.constants,
            ..Default::default()
        },
    }
}
pub fn vs_main_none_entry() -> VertexEntry<0> {
    VertexEntry {
        entry_point: ENTRY_VS_MAIN_NONE,
        buffers: [],
        constants: Default::default(),
    }
}
pub fn vs_main_single_entry() -> VertexEntry<0> {
    VertexEntry {
        entry_point: ENTRY_VS_MAIN_SINGLE,
        buffers: [],
        constants: Default::default(),
    }
}
pub fn vs_main_multiple_entry() -> VertexEntry<0> {
    VertexEntry {
        entry_point: ENTRY_VS_MAIN_MULTIPLE,
        buffers: [],
        constants: Default::default(),
    }
}
pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let source = std::borrow::Cow::Borrowed(include_str!("shader.wgsl"));
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source),
    })
}
pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    })
}
