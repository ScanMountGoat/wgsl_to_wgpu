#[derive(Debug)]
pub struct FragmentEntry<const N: usize> {
    pub entry_point: &'static str,
    pub targets: [Option<wgpu::ColorTargetState>; N],
    pub constants: Vec<(&'static str, f64)>,
}
pub fn fragment_state<'a, const N: usize>(
    module: &'a wgpu::ShaderModule,
    entry: &'a FragmentEntry<N>,
) -> wgpu::FragmentState<'a> {
    wgpu::FragmentState {
        module,
        entry_point: Some(entry.entry_point),
        targets: &entry.targets,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &entry.constants,
            ..Default::default()
        },
    }
}
pub fn main_entry(targets: [Option<wgpu::ColorTargetState>; 0]) -> FragmentEntry<0> {
    FragmentEntry {
        entry_point: ENTRY_MAIN,
        targets,
        constants: Default::default(),
    }
}
pub const SOURCE: &str = include_str!("types.wgsl");
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
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct Atomics {
    pub num: u32,
    pub numi: i32,
}
pub const ENTRY_MAIN: &str = "main";
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct MatricesF16 {
    pub a: [[half::f16; 4]; 4],
    pub b: [[half::f16; 4]; 3],
    pub c: [[half::f16; 4]; 2],
    pub d: [[half::f16; 3]; 4],
    pub e: [[half::f16; 3]; 3],
    pub f: [[half::f16; 3]; 2],
    pub g: [[half::f16; 2]; 4],
    pub h: [[half::f16; 2]; 3],
    pub i: [[half::f16; 2]; 2],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
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
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
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
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct Nested {
    pub a: MatricesF32,
    pub b: MatricesF64,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct Scalars {
    pub a: u32,
    pub b: i32,
    pub c: f32,
    pub d: half::f16,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct StaticArrays {
    pub a: [u32; 5],
    pub b: [f32; 3],
    pub c: [[[f32; 4]; 4]; 512],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct VectorsF16 {
    pub a: [half::f16; 2],
    pub b: [half::f16; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct VectorsF32 {
    pub a: [f32; 2],
    pub b: [f32; 3],
    pub c: [f32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct VectorsF64 {
    pub a: [f64; 2],
    pub b: [f64; 3],
    pub c: [f64; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct VectorsI32 {
    pub a: [i32; 2],
    pub b: [i32; 3],
    pub c: [i32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
pub struct VectorsU32 {
    pub a: [u32; 2],
    pub b: [u32; 3],
    pub c: [u32; 4],
}
