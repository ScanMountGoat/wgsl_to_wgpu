pub mod bind_groups {
    #[derive(Debug, Clone)]
    pub struct BindGroup0(wgpu::BindGroup);
    #[derive(Debug)]
    pub struct BindGroupLayout0<'a> {
        pub uniforms: wgpu::BufferBinding<'a>,
    }
    const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
        label: Some("LayoutDescriptor0"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    };
    impl BindGroup0 {
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
        }
        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(bindings.uniforms),
                }],
                label: Some("BindGroup0"),
            });
            Self(bind_group)
        }
        pub fn set<P: super::SetBindGroup>(&self, pass: &mut P) {
            pass.set_bind_group(0, &self.0, &[]);
        }
        pub fn inner(&self) -> &wgpu::BindGroup {
            &self.0
        }
    }
    #[derive(Debug, Copy, Clone)]
    pub struct BindGroups<'a> {
        pub bind_group0: &'a BindGroup0,
    }
    impl BindGroups<'_> {
        pub fn set<P: super::SetBindGroup>(&self, pass: &mut P) {
            self.bind_group0.set(pass);
        }
    }
}
pub fn set_bind_groups<P: SetBindGroup>(pass: &mut P, bind_group0: &bind_groups::BindGroup0) {
    bind_group0.set(pass);
}
pub fn frag_entry(targets: [Option<wgpu::ColorTargetState>; 1]) -> FragmentEntry<1> {
    FragmentEntry {
        entry_point: ENTRY_FRAG,
        targets,
        constants: Default::default(),
    }
}
pub const SOURCE : & str = "struct uniforms_Uniforms {\n    a: vec4<f32>,\n    b: uniforms_nested_Nested,\n}\n\nstruct uniforms_Uniforms2 {\n    b: uniforms_nested_Nested,\n}\n\nstruct uniforms_nested_Nested {\n    a: vec4<f32>,\n    b: Root,\n    c: shared_Shared\n}\n\nstruct Root {\n    c: vec4<f32>\n}\n\nstruct shared_Shared {\n    d: vec4<f32>\n}\n\nstruct shared_VertexInput {\n    @location(0) position: vec3<f32>\n}\n\nstruct VertexOffset {\n    @location(1) offset: vec3<f32>\n}\n\nstruct shared_VertexOutput {\n    @builtin(position) clip_position: vec4<f32>,\n}\n\nconst shared_TEST: f32 = 1.0;\n\n@group(0) @binding(0)\nvar<uniform> bindings_uniforms: uniforms_Uniforms;\n\n@vertex\nfn vert(in: shared_VertexInput, offset: VertexOffset) -> shared_VertexOutput {\n    var out: shared_VertexOutput;\n    out.clip_position = vec4(in.position + offset.offset, shared_TEST);\n    return out;\n}\n\n@vertex\nfn shared_vert(in: shared_VertexInput, offset: VertexOffset) -> shared_VertexOutput {\n    var out: shared_VertexOutput;\n    out.clip_position = vec4(in.position + offset.offset, shared_TEST);\n    return out;\n}\n\n@fragment\nfn frag(in: shared_VertexOutput) -> @location(0) vec4<f32> {\n    return bindings_uniforms.b.a * vec4(0.0);\n}" ;
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
        bind_group_layouts: &[&bind_groups::BindGroup0::get_bind_group_layout(device)],
        immediate_size: 0,
    })
}
pub const ENTRY_FRAG: &str = "frag";
pub const ENTRY_VERT: &str = "vert";
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Root {
    pub c: [f32; 4],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VertexOffset {
    pub offset: [f32; 3],
}
impl VertexOffset {
    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: std::mem::offset_of!(VertexOffset, offset) as u64,
        shader_location: 1,
    }];
    pub const fn vertex_buffer_layout(
        step_mode: wgpu::VertexStepMode,
    ) -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexOffset>() as u64,
            step_mode,
            attributes: &VertexOffset::VERTEX_ATTRIBUTES,
        }
    }
}
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
pub trait SetBindGroup {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    );
}
impl SetBindGroup for wgpu::ComputePass<'_> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets);
    }
}
impl SetBindGroup for wgpu::RenderPass<'_> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets);
    }
}
impl SetBindGroup for wgpu::RenderBundleEncoder<'_> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets);
    }
}
pub fn vert_entry(
    vertex_input: wgpu::VertexStepMode,
    vertex_offset: wgpu::VertexStepMode,
) -> VertexEntry<2> {
    VertexEntry {
        entry_point: ENTRY_VERT,
        buffers: [
            shared::VertexInput::vertex_buffer_layout(vertex_input),
            VertexOffset::vertex_buffer_layout(vertex_offset),
        ],
        constants: Default::default(),
    }
}
pub mod shared {
    pub const ENTRY_VERT: &str = "shared_vert";
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Shared {
        pub d: [f32; 4],
    }
    pub const TEST: f32 = 1f32;
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct VertexInput {
        pub position: [f32; 3],
    }
    impl VertexInput {
        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: std::mem::offset_of!(VertexInput, position) as u64,
            shader_location: 0,
        }];
        pub const fn vertex_buffer_layout(
            step_mode: wgpu::VertexStepMode,
        ) -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<VertexInput>() as u64,
                step_mode,
                attributes: &VertexInput::VERTEX_ATTRIBUTES,
            }
        }
    }
    pub fn vert_entry(
        vertex_input: wgpu::VertexStepMode,
        vertex_offset: wgpu::VertexStepMode,
    ) -> super::VertexEntry<2> {
        super::VertexEntry {
            entry_point: ENTRY_VERT,
            buffers: [
                VertexInput::vertex_buffer_layout(vertex_input),
                super::VertexOffset::vertex_buffer_layout(vertex_offset),
            ],
            constants: Default::default(),
        }
    }
}
pub mod uniforms {
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Uniforms {
        pub a: [f32; 4],
        pub b: nested::Nested,
    }
    pub mod nested {
        #[repr(C)]
        #[derive(Debug, Copy, Clone, PartialEq)]
        pub struct Nested {
            pub a: [f32; 4],
            pub b: super::super::Root,
            pub c: super::super::shared::Shared,
        }
    }
}
