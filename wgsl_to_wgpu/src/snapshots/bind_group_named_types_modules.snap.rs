pub fn set_bind_groups<P: SetBindGroup>(
    pass: &mut P,
    bind_group_camera_Settings: &bind_groups::BindGroupCameraSettings,
) {
    bind_group_camera_Settings.set(pass);
}
pub const SOURCE : & str = "struct shared_Camera {\n\tmatrix: mat4x4<f32>,\n}\n\nstruct settings_Settings {\n\tsomethings: mat4x4<f32>,\n}\n\n@group(0) @binding(0)\nvar shared_camera: shared_Camera;\n@group(0) @binding(1)\nvar shared_Settings: settings_Settings;\n" ;
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
        bind_group_layouts: &[&bind_groups::BindGroupCameraSettings::get_bind_group_layout(device)],
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
pub mod bind_groups {
    #[derive(Debug, Clone)]
    pub struct BindGroupCameraSettings(wgpu::BindGroup);
    #[derive(Debug)]
    pub struct BindGroupLayoutCameraSettings<'a> {
        pub camera: wgpu::BufferBinding<'a>,
        pub Settings: wgpu::BufferBinding<'a>,
    }
    const LAYOUT_DESCRIPTOR_CAMERA_SETTINGS: wgpu::BindGroupLayoutDescriptor =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("LayoutDescriptorCameraSettings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::NONE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::NONE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };
    impl BindGroupCameraSettings {
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_CAMERA_SETTINGS)
        }
        pub fn from_bindings(
            device: &wgpu::Device,
            bindings: BindGroupLayoutCameraSettings,
        ) -> Self {
            let bind_group_layout =
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_CAMERA_SETTINGS);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(bindings.camera),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(bindings.Settings),
                    },
                ],
                label: Some("BindGroupCameraSettings"),
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
        pub bind_group_camera_Settings: &'a BindGroupCameraSettings,
    }
    impl BindGroups<'_> {
        pub fn set<P: super::SetBindGroup>(&self, pass: &mut P) {
            self.bind_group_camera_Settings.set(pass);
        }
    }
}
pub mod settings {
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Settings {
        pub somethings: [[f32; 4]; 4],
    }
}
pub mod shared {
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Camera {
        pub matrix: [[f32; 4]; 4],
    }
}
