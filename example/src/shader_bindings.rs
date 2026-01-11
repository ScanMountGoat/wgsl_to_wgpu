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
pub mod compute_shader {
    pub mod bind_groups {
        #[derive(Debug, Clone)]
        pub struct BindGroupUniformsInvocationCount(wgpu::BindGroup);
        #[derive(Debug)]
        pub struct BindGroupLayoutUniformsInvocationCount<'a> {
            pub uniforms: wgpu::BufferBinding<'a>,
            pub invocation_count: wgpu::BufferBinding<'a>,
        }
        const LAYOUT_DESCRIPTOR_UNIFORMS_INVOCATION_COUNT: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptorUniformsInvocationCount"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            };
        impl BindGroupUniformsInvocationCount {
            pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_UNIFORMS_INVOCATION_COUNT)
            }
            pub fn from_bindings(
                device: &wgpu::Device,
                bindings: BindGroupLayoutUniformsInvocationCount,
            ) -> Self {
                let bind_group_layout =
                    device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_UNIFORMS_INVOCATION_COUNT);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(bindings.uniforms),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(bindings.invocation_count),
                        },
                    ],
                    label: Some("BindGroupUniformsInvocationCount"),
                });
                Self(bind_group)
            }
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                pass.set_bind_group(0, &self.0, &[]);
            }
            pub fn inner(&self) -> &wgpu::BindGroup {
                &self.0
            }
        }
        #[derive(Debug, Copy, Clone)]
        pub struct BindGroups<'a> {
            pub bind_group_uniforms_invocation_count: &'a BindGroupUniformsInvocationCount,
        }
        impl BindGroups<'_> {
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                self.bind_group_uniforms_invocation_count.set(pass);
            }
        }
    }
    pub fn set_bind_groups<P: super::SetBindGroup>(
        pass: &mut P,
        bind_group_uniforms_invocation_count: &bind_groups::BindGroupUniformsInvocationCount,
    ) {
        bind_group_uniforms_invocation_count.set(pass);
    }
    pub mod compute {
        pub const MAIN_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];
        pub fn create_main_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
            let module = super::create_shader_module(device);
            let layout = super::create_pipeline_layout(device);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline main"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            })
        }
    }
    pub const SOURCE: &str = include_str!("compute_shader.wgsl");
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
            bind_group_layouts: &[
                &bind_groups::BindGroupUniformsInvocationCount::get_bind_group_layout(device),
            ],
            immediate_size: 0,
        })
    }
    pub const ENTRY_MAIN: &str = "main";
}
pub mod shader {
    pub struct OverrideConstants {
        pub force_black: bool,
        pub scale: Option<f32>,
    }
    impl OverrideConstants {
        pub fn constants(&self) -> Vec<(&'static str, f64)> {
            let mut entries = vec![("force_black", if self.force_black { 1.0 } else { 0.0 })];
            if let Some(value) = self.scale {
                entries.push(("scale", value as f64));
            }
            entries
        }
    }
    pub mod bind_groups {
        #[derive(Debug, Clone)]
        pub struct BindGroupColorTextureColorSampler(wgpu::BindGroup);
        #[derive(Debug)]
        pub struct BindGroupLayoutColorTextureColorSampler<'a> {
            pub color_texture: &'a wgpu::TextureView,
            pub color_sampler: &'a wgpu::Sampler,
        }
        const LAYOUT_DESCRIPTOR_COLOR_TEXTURE_COLOR_SAMPLER: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptorColorTextureColorSampler"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            };
        impl BindGroupColorTextureColorSampler {
            pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_COLOR_TEXTURE_COLOR_SAMPLER)
            }
            pub fn from_bindings(
                device: &wgpu::Device,
                bindings: BindGroupLayoutColorTextureColorSampler,
            ) -> Self {
                let bind_group_layout =
                    device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_COLOR_TEXTURE_COLOR_SAMPLER);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(bindings.color_texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(bindings.color_sampler),
                        },
                    ],
                    label: Some("BindGroupColorTextureColorSampler"),
                });
                Self(bind_group)
            }
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                pass.set_bind_group(0, &self.0, &[]);
            }
            pub fn inner(&self) -> &wgpu::BindGroup {
                &self.0
            }
        }
        #[derive(Debug, Copy, Clone)]
        pub struct BindGroups<'a> {
            pub bind_group_color_texture_color_sampler: &'a BindGroupColorTextureColorSampler,
            pub bind_group_uniforms: &'a super::super::shared::BindGroupUniforms,
        }
        impl BindGroups<'_> {
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                self.bind_group_color_texture_color_sampler.set(pass);
                self.bind_group_uniforms.set(pass, 1);
            }
        }
    }
    pub fn set_bind_groups<P: super::SetBindGroup>(
        pass: &mut P,
        bind_group_color_texture_color_sampler: &bind_groups::BindGroupColorTextureColorSampler,
        bind_group_uniforms: &super::shared::BindGroupUniforms,
    ) {
        bind_group_color_texture_color_sampler.set(pass);
        bind_group_uniforms.set(pass, 1);
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
    pub fn fs_main_entry(
        targets: [Option<wgpu::ColorTargetState>; 1],
        overrides: &OverrideConstants,
    ) -> FragmentEntry<1> {
        FragmentEntry {
            entry_point: ENTRY_FS_MAIN,
            targets,
            constants: overrides.constants(),
        }
    }
    pub const SOURCE: &str = include_str!("shader.wgsl");
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
            bind_group_layouts: &[
                &bind_groups::BindGroupColorTextureColorSampler::get_bind_group_layout(device),
                &super::shared::BindGroupUniforms::get_bind_group_layout(device),
            ],
            immediate_size: 64,
        })
    }
    pub const ENTRY_FS_MAIN: &str = "fs_main";
    pub const ENTRY_VS_MAIN: &str = "vs_main";
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
    pub struct PushConstants {
        pub color_matrix: glam::Mat4,
    }
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, bytemuck :: Pod, bytemuck :: Zeroable)]
    pub struct VertexInput {
        pub position: glam::Vec3,
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
    pub fn vs_main_entry(
        vertex_input: wgpu::VertexStepMode,
        overrides: &OverrideConstants,
    ) -> super::VertexEntry<1> {
        super::VertexEntry {
            entry_point: ENTRY_VS_MAIN,
            buffers: [VertexInput::vertex_buffer_layout(vertex_input)],
            constants: overrides.constants(),
        }
    }
}
pub mod shader_triangle {
    pub mod bind_groups {
        #[derive(Debug, Copy, Clone)]
        pub struct BindGroups<'a> {
            pub bind_group_uniforms: &'a super::super::shared::BindGroupUniforms,
        }
        impl BindGroups<'_> {
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                self.bind_group_uniforms.set(pass, 0);
            }
        }
    }
    pub fn set_bind_groups<P: super::SetBindGroup>(
        pass: &mut P,
        bind_group_uniforms: &super::shared::BindGroupUniforms,
    ) {
        bind_group_uniforms.set(pass, 0);
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
    pub fn fs_main_entry(targets: [Option<wgpu::ColorTargetState>; 1]) -> FragmentEntry<1> {
        FragmentEntry {
            entry_point: ENTRY_FS_MAIN,
            targets,
            constants: Default::default(),
        }
    }
    pub const SOURCE: &str = include_str!("shader_triangle.wgsl");
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
            bind_group_layouts: &[&super::shared::BindGroupUniforms::get_bind_group_layout(
                device,
            )],
            immediate_size: 0,
        })
    }
    pub const ENTRY_FS_MAIN: &str = "fs_main";
    pub const ENTRY_VS_MAIN: &str = "vs_main";
    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, bytemuck :: Pod, bytemuck :: Zeroable)]
    pub struct VertexInput {
        pub position: glam::Vec3,
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
    pub fn vs_main_entry(vertex_input: wgpu::VertexStepMode) -> super::VertexEntry<1> {
        super::VertexEntry {
            entry_point: ENTRY_VS_MAIN,
            buffers: [VertexInput::vertex_buffer_layout(vertex_input)],
            constants: Default::default(),
        }
    }
}
pub mod shared {
    #[derive(Debug, Clone)]
    pub struct BindGroupUniforms(wgpu::BindGroup);
    #[derive(Debug)]
    pub struct BindGroupLayoutUniforms<'a> {
        pub uniforms: wgpu::BufferBinding<'a>,
    }
    const LAYOUT_DESCRIPTOR_UNIFORMS: wgpu::BindGroupLayoutDescriptor =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("LayoutDescriptorUniforms"),
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
    impl BindGroupUniforms {
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_UNIFORMS)
        }
        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayoutUniforms) -> Self {
            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR_UNIFORMS);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(bindings.uniforms),
                }],
                label: Some("BindGroupUniforms"),
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
    #[derive(Debug, Copy, Clone, PartialEq, encase :: ShaderType)]
    pub struct Uniforms {
        pub color_rgb: glam::Vec3,
    }
}
