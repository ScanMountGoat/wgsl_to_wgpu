pub mod bind_groups {
    #[derive(Debug, Copy, Clone)]
    pub struct BindGroups<'a> {
        pub bind_group_camera: &'a super::shared::BindGroupCamera,
    }
    impl BindGroups<'_> {
        pub fn set<P: super::SetBindGroup>(&self, pass: &mut P) {
            self.bind_group_camera.set(pass, 0);
        }
    }
}
pub fn set_bind_groups<P: SetBindGroup>(pass: &mut P, bind_group_camera: &shared::BindGroupCamera) {
    bind_group_camera.set(pass, 0);
}
pub const SOURCE : & str = "struct shared_Camera {\n\tmatrix: mat4x4<f32>,\n}\n\n@group(0) @binding(0)\nvar<uniform> shared_camera : shared_Camera;\n" ;
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
        bind_group_layouts: &[&shared::BindGroupCamera::get_bind_group_layout(device)],
        immediate_size: 0,
    })
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
pub mod shared {
    #[derive(Debug, Clone)]
    pub struct BindGroupCamera(wgpu::BindGroup);
    #[derive(Debug)]
    pub struct BindGroupLayoutCamera<'a> {
        pub camera: wgpu::BufferBinding<'a>,
    }
    const LAYOUT_DESCRIPTOR_CAMERA: wgpu::BindGroupLayoutDescriptor =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("LayoutDescriptorCamera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::NONE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        };
    impl BindGroupCamera {
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_CAMERA)
        }
        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayoutCamera) -> Self {
            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_CAMERA);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(bindings.camera),
                }],
                label: Some("BindGroupCamera"),
            });
            Self(bind_group)
        }
        pub fn set<P: super::SetBindGroup>(&self, pass: &mut P, index: u32) {
            pass.set_bind_group(index, &self.0, &[]);
        }
        pub fn inner(&self) -> &wgpu::BindGroup {
            &self.0
        }
    }
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Camera {
        pub matrix: [[f32; 4]; 4],
    }
}
