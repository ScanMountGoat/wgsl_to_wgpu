pub mod bind_groups {
    #[derive(Debug, Copy, Clone)]
    pub struct BindGroups<'a> {
        pub bind_group_texture_sampler: &'a super::atlas::BindGroupTextureSampler,
    }
    impl BindGroups<'_> {
        pub fn set<P: super::SetBindGroup>(&self, pass: &mut P) {
            self.bind_group_texture_sampler.set(pass, 0);
        }
    }
}
pub fn set_bind_groups<P: SetBindGroup>(
    pass: &mut P,
    bind_group_texture_sampler: &atlas::BindGroupTextureSampler,
) {
    bind_group_texture_sampler.set(pass, 0);
}
pub const SOURCE : & str = "@group(0) @binding(0)\nvar atlas_texture: texture_2d<f32>;\n@group(0) @binding(1)\nvar atlas_sampler: sampler;\n" ;
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
        bind_group_layouts: &[&atlas::BindGroupTextureSampler::get_bind_group_layout(
            device,
        )],
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
pub mod atlas {
    #[derive(Debug, Clone)]
    pub struct BindGroupTextureSampler(wgpu::BindGroup);
    #[derive(Debug)]
    pub struct BindGroupLayoutTextureSampler<'a> {
        pub texture: &'a wgpu::TextureView,
        pub sampler: &'a wgpu::Sampler,
    }
    const LAYOUT_DESCRIPTOR_TEXTURE_SAMPLER: wgpu::BindGroupLayoutDescriptor =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("LayoutDescriptorTextureSampler"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::NONE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::NONE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        };
    impl BindGroupTextureSampler {
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_TEXTURE_SAMPLER)
        }
        pub fn from_bindings(
            device: &wgpu::Device,
            bindings: BindGroupLayoutTextureSampler,
        ) -> Self {
            let bind_group_layout =
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_TEXTURE_SAMPLER);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(bindings.texture),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(bindings.sampler),
                    },
                ],
                label: Some("BindGroupTextureSampler"),
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
}
