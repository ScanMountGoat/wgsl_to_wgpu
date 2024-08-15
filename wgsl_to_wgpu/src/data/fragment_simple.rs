pub const ENTRY_FS_MAIN: &str = "fs_main";
#[derive(Debug)]
pub struct FragmentEntry<const N: usize> {
    pub entry_point: &'static str,
    pub targets: [Option<wgpu::ColorTargetState>; N],
    pub constants: std::collections::HashMap<String, f64>,
}
pub fn fragment_state<'a, const N: usize>(
    module: &'a wgpu::ShaderModule,
    entry: &'a FragmentEntry<N>,
) -> wgpu::FragmentState<'a> {
    wgpu::FragmentState {
        module,
        entry_point: entry.entry_point,
        targets: &entry.targets,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &entry.constants,
            ..Default::default()
        },
    }
}
pub fn fs_main_entry(targets: [Option<wgpu::ColorTargetState>; 0]) -> FragmentEntry<0> {
    FragmentEntry {
        entry_point: ENTRY_FS_MAIN,
        targets,
        constants: Default::default(),
    }
}
pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let source = std::borrow::Cow::Borrowed("@fragment\nfn fs_main() {}\n");
    device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source),
        })
}
pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device
        .create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            },
        )
}
