//! # wgsl_to_wgpu
//! wgsl_to_wgpu is an experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Features
//! The [write_module_file] function is intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! ## Limitations
//! This project currently supports a small subset of WGSL types and doesn't enforce certain key properties such as field alignment.
//! It's recommended for now to only run the Rust file generation as needed and rely on WGPU's runtime validation to fix any potential errors.
use indoc::{formatdoc, writedoc};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

mod wgsl;

// TODO: Simplify these templates and indentation?
// TODO: Structure the code to make it easier to imagine what the output will look like.

/// Parses the WGSL shader from `wgsl_path` and returns the generated Rust module's source code as a [String].
/// The `wgsl_include_path` should be a valid path for the `include_wgsl!` macro used in the generated file.
pub fn create_shader_module<P: AsRef<Path>>(wgsl_path: P, wgsl_include_path: &str) -> String {
    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl_source).unwrap();
    let bind_group_data = wgsl::get_bind_group_data(&module);

    let mut output = String::new();
    let shader_stages = wgsl::shader_stages(&module);

    // Write all the structs, including uniforms and entry function inputs.
    write_structs(&mut output, 0, &module);

    // TODO: Avoid having a dependency on naga here?
    write_bind_groups_module(&mut output, &bind_group_data, shader_stages);
    write_vertex_module(&mut output, &module);

    writedoc!(
        output,
        r#"
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {{
                device.create_shader_module(&wgpu::ShaderModuleDescriptor {{
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("{wgsl_include_path}")))
                }})
            }}
        "#
    )
    .unwrap();

    // TODO: Find a cleaner way of doing this?
    let bind_group_layouts = bind_group_data
        .iter()
        .map(|(group_no, _)| {
            format!("&bind_groups::BindGroup{group_no}::get_bind_group_layout(device),")
        })
        .collect::<Vec<String>>()
        .join("\n            ");

    writedoc!(
        output,
        r#"
            pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {{
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {{
                    label: None,
                    bind_group_layouts: &[
                        {bind_group_layouts}
                    ],
                    push_constant_ranges: &[],
                }})
            }}
        "#
    )
    .unwrap();

    output
}

// Apply indentation to each level.
fn indent(str: String, level: usize) -> String {
    str.lines()
        .map(|l| " ".repeat(level) + l)
        .collect::<Vec<String>>()
        .join("\n")
}

// Assume the input is already unindented with indoc.
fn write_indented<W: Write>(w: &mut W, level: usize, str: String) {
    writeln!(w, "{}", indent(str, level)).unwrap();
}

fn write_vertex_module<W: Write>(f: &mut W, module: &naga::Module) {
    writeln!(f, "pub mod vertex {{").unwrap();

    write_attribute_locations(f, module);
    // TODO: This is redundant with above?
    write_vertex_input_structs(f, module);

    writeln!(f, "}}").unwrap();
}

fn write_attribute_locations<W: Write>(f: &mut W, module: &naga::Module) {
    // TODO: Should these be part of each struct?
    // TODO: Generate the vertex state for each input?
    for (name, location) in wgsl::get_vertex_input_locations(&module) {
        // TODO: Use const case
        let const_name = name.to_uppercase();
        writeln!(
            f,
            "    pub const {const_name}_LOCATION: u32 = {location}u32;"
        )
        .unwrap();
    }
}

// TODO: Test this?
fn write_vertex_input_structs<W: Write>(f: &mut W, module: &naga::Module) {
    let vertex_inputs = wgsl::get_vertex_input_structs(module);
    for input in vertex_inputs {
        let name = input.name;

        let count = input.fields.len();
        let attributes = input
            .fields
            .iter()
            .map(|(location, m)| {
                let format = wgsl::vertex_format(&module.types[m.ty]);
                // TODO: Will the debug implementation always work with the macro?
                format!("{location} => {:?}", format)
            })
            .collect::<Vec<_>>()
            .join(", ");

        // TODO: Account for alignment/padding?
        let size_in_bytes: u64 = input
            .fields
            .iter()
            .map(|(_, m)| wgsl::vertex_format(&module.types[m.ty]).size())
            .sum();

        // The vertex input structs should already be written at this point.
        // TODO: Support vertex inputs that aren't in a struct.
        write_indented(
            f,
            4,
            formatdoc!(
                r#"
                    impl super::{name} {{
                        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; {count}] = wgpu::vertex_attr_array![{attributes}];
                        /// The total size in bytes of all fields without considering padding or alignment.
                        pub const SIZE_IN_BYTES: u64 = {size_in_bytes};
                    }}
                "#
            ),
        );
    }
}

// TODO: Take an iterator instead?
fn write_bind_groups_module<W: Write>(
    f: &mut W,
    bind_group_data: &BTreeMap<u32, wgsl::GroupData>,
    shader_stages: wgpu::ShaderStages,
) {
    writeln!(f, "pub mod bind_groups {{").unwrap();

    for (group_no, group) in bind_group_data {
        writeln!(f, "    pub struct BindGroup{group_no}(wgpu::BindGroup);").unwrap();

        write_bind_group_layout(f, 4, *group_no, &group);
        write_bind_group_layout_descriptor(f, 4, *group_no, &group, shader_stages);
        impl_bind_group(f, 4, *group_no, &group, shader_stages);
    }

    writeln!(f, "    pub struct BindGroups<'a> {{").unwrap();
    for (group_no, _) in bind_group_data {
        writeln!(
            f,
            "        pub bind_group{group_no}: &'a BindGroup{group_no},"
        )
        .unwrap();
    }
    writeln!(f, "    }}").unwrap();

    // TODO: Test this?
    // TODO: Support compute shader with vertex/fragment in the same module?
    let is_compute = shader_stages == wgpu::ShaderStages::COMPUTE;

    let render_pass = if is_compute {
        "wgpu::ComputePass<'a>"
    } else {
        "wgpu::RenderPass<'a>"
    };

    writeln!(f, "    pub fn set_bind_groups<'a>(").unwrap();
    writeln!(f, "        pass: &mut {render_pass},").unwrap();
    writeln!(f, "        bind_groups: BindGroups<'a>,").unwrap();
    writeln!(f, "    ) {{").unwrap();
    for (group_no, _) in bind_group_data {
        writeln!(
            f,
            "        pass.set_bind_group({group_no}u32, &bind_groups.bind_group{group_no}.0, &[]);"
        )
        .unwrap();
    }
    writeln!(f, "    }}").unwrap();

    writeln!(f, "}}").unwrap();
}

fn write_structs<W: Write>(f: &mut W, indent: usize, module: &naga::Module) {
    // Create matching Rust structs for WGSL structs.
    // The goal is to eventually have safe ways to initialize uniform buffers.

    // TODO: How to provide a convenient way to work with these types.
    // Users will want to either a) create a new buffer each type or b) reuse an existing buffer.
    // It might not make sense from a performance perspective to constantly create new resources.
    // This requires the user to keep track of the buffer separately from the BindGroup itself.

    // This is a UniqueArena, so types will only be defined once.
    for (_, t) in module.types.iter() {
        match &t.inner {
            naga::TypeInner::Struct { members, .. } => {
                let name = t.name.as_ref().unwrap();
                // TODO: Enforce std140 with crevice for uniform buffers to be safe?
                write_indented(
                    f,
                    indent,
                    formatdoc!(
                        r"
                        #[repr(C)]
                        #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                        pub struct {name} {{
                        "
                    ),
                );

                write_struct_members(f, indent + 4, members, module);
                write_indented(f, indent, formatdoc!("}}"));
            }
            _ => (),
        }
    }
}

fn write_struct_members<W: Write>(
    f: &mut W,
    indent: usize,
    members: &[naga::StructMember],
    module: &naga::Module,
) {
    for member in members {
        let member_name = member.name.as_ref().unwrap();
        let member_type = wgsl::rust_type(&module, &module.types[member.ty]);
        write_indented(f, indent, formatdoc!("pub {member_name}: {member_type},"));
    }
}

fn write_bind_group_layout<W: Write>(
    f: &mut W,
    indent: usize,
    group_no: u32,
    group: &wgsl::GroupData,
) {
    write_indented(
        f,
        indent,
        formatdoc!("pub struct BindGroupLayout{group_no}<'a> {{"),
    );
    for binding in &group.bindings {
        let field_name = binding.name.as_ref().unwrap();
        // TODO: Support more types.
        let field_type = match binding.binding_type.inner {
            // TODO: Is it possible to make structs strongly typed and handle buffer creation automatically?
            // This could be its own module and associated tests.
            naga::TypeInner::Struct { .. } => "wgpu::Buffer",
            naga::TypeInner::Image { .. } => "wgpu::TextureView",
            naga::TypeInner::Sampler { .. } => "wgpu::Sampler",
            _ => panic!("Unsupported type for binding fields."),
        };
        write_indented(
            f,
            indent + 4,
            formatdoc!("pub {field_name}: &'a {field_type},"),
        );
    }
    write_indented(f, indent, formatdoc!("}}"));
}

fn write_bind_group_layout_descriptor<W: Write>(
    f: &mut W,
    indent: usize,
    group_no: u32,
    group: &wgsl::GroupData,
    shader_stages: wgpu::ShaderStages,
) {
    write_indented(
        f,
        indent,
        formatdoc!(
            r#"
                const LAYOUT_DESCRIPTOR{group_no}: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {{
                    label: None,
                    entries: &[
            "#
        ),
    );
    for binding in &group.bindings {
        write_bind_group_layout_entry(f, binding, indent + 8, shader_stages);
    }
    write_indented(
        f,
        indent,
        formatdoc!(
            r#"
                    ]
                }};
            "#
        ),
    );
}

fn write_bind_group_layout_entry<W: Write>(
    f: &mut W,
    binding: &wgsl::GroupBinding,
    indent: usize,
    shader_stages: wgpu::ShaderStages,
) {
    // TODO: Assume storage is only used for compute?
    // TODO: Support just vertex or fragment?
    // TODO: Visible from all stages?
    let stages = match shader_stages {
        wgpu::ShaderStages::VERTEX_FRAGMENT => "wgpu::ShaderStages::VERTEX_FRAGMENT",
        wgpu::ShaderStages::COMPUTE => "wgpu::ShaderStages::COMPUTE",
        _ => todo!(),
    };

    let binding_index = binding.binding_index;
    write_indented(
        f,
        indent,
        formatdoc!(
            r#"
                wgpu::BindGroupLayoutEntry {{
                    binding: {binding_index}u32,
                    visibility: {stages},
            "#
        ),
    );
    // TODO: Support more types.
    match binding.binding_type.inner {
        naga::TypeInner::Struct { .. } => {
            let buffer_binding_type = wgsl::buffer_binding_type(binding.storage_class);
            write_indented(
                f,
                indent + 4,
                formatdoc!(
                    r#"
                        ty: wgpu::BindingType::Buffer {{
                            ty: {buffer_binding_type},
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }},
                    "#
                ),
            );
        }
        naga::TypeInner::Image { dim, .. } => {
            let view_dim = match dim {
                naga::ImageDimension::D1 => "wgpu::TextureViewDimension::D1",
                naga::ImageDimension::D2 => "wgpu::TextureViewDimension::D2",
                naga::ImageDimension::D3 => "wgpu::TextureViewDimension::D3",
                naga::ImageDimension::Cube => "wgpu::TextureViewDimension::Cube",
            };

            write_indented(
                f,
                indent + 4,
                formatdoc!(
                    r#"
                        ty: wgpu::BindingType::Texture {{
                            multisampled: false,
                            view_dimension: {view_dim},
                            sample_type: wgpu::TextureSampleType::Float {{ filterable: true }},
                        }},
                    "#
                ),
            );
        }
        naga::TypeInner::Sampler { .. } => {
            // TODO: Don't assume filtering?
            write_indented(
                f,
                indent + 4,
                "ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),".to_string(),
            );
        }
        // TODO: Better error handling.
        _ => panic!("Failed to generate BindingType."),
    };
    write_indented(
        f,
        indent,
        formatdoc!(
            r#"
                    count: None,
                }},
            "#
        ),
    );
}

fn impl_bind_group<W: Write>(
    f: &mut W,
    indent: usize,
    group_no: u32,
    group: &wgsl::GroupData,
    shader_stages: wgpu::ShaderStages,
) {
    write_indented(
        f,
        indent,
        formatdoc!(
            r#"
                impl BindGroup{group_no} {{
                    pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {{
                        device.create_bind_group_layout(&LAYOUT_DESCRIPTOR{group_no})
                    }}

                    pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout{group_no}) -> Self {{
                        let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR{group_no});
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {{
                            layout: &bind_group_layout,
                            entries: &[
            "#
        ),
    );

    for binding in &group.bindings {
        let binding_index = binding.binding_index;
        let binding_name = binding.name.as_ref().unwrap();
        let resource_type = match binding.binding_type.inner {
            naga::TypeInner::Struct { .. } => {
                // TODO: Don't assume the entire buffer is used.
                format!("bindings.{binding_name}.as_entire_binding()")
            }
            naga::TypeInner::Image { .. } => {
                format!("wgpu::BindingResource::TextureView(bindings.{binding_name})")
            }
            naga::TypeInner::Sampler { .. } => {
                format!("wgpu::BindingResource::Sampler(bindings.{binding_name})")
            }
            // TODO: Better error handling.
            _ => panic!("Failed to generate BindingType."),
        };

        write_indented(
            f,
            indent + 16,
            formatdoc!(
                r#"
                    wgpu::BindGroupEntry {{
                        binding: {binding_index}u32,
                        resource: {resource_type},
                    }},
                "#
            ),
        );
    }
    write_indented(
        f,
        indent + 4,
        formatdoc!(
            r#"
                        ],
                        label: None,
                    }});
                    Self(bind_group)
                }}
            "#
        ),
    );

    // TODO: Test this?
    // TODO: Support compute shader with vertex/fragment in the same module?
    let is_compute = shader_stages == wgpu::ShaderStages::COMPUTE;

    let render_pass = if is_compute {
        "wgpu::ComputePass<'a>"
    } else {
        "wgpu::RenderPass<'a>"
    };

    write_indented(
        f,
        indent,
        formatdoc!(
            r#"

                pub fn set<'a>(&'a self, render_pass: &mut {render_pass}) {{
                    render_pass.set_bind_group({group_no}u32, &self.0, &[]);
                }}
            }}"#
        ),
    );
}

#[cfg(test)]
mod test {
    use super::*;
    use indoc::indoc;

    #[test]
    fn write_all_structs() {
        // TODO: Test different types.
        let source = indoc! {r#"
            struct VectorsF32 {
                a: vec2<f32>;
                b: vec3<f32>;
                c: vec4<f32>;
            };

            struct VectorsU32 {
                a: vec2<u32>;
                b: vec3<u32>;
                c: vec4<u32>;
            };

            struct MatricesF32 {
                a: mat4x4<f32>;
            };
            
            struct StaticArrays {
                a: array<u32, 5>;
                b: array<f32, 3>;
                c: array<mat4x4<f32>, 512>;
            };

            [[stage(fragment)]]
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();

        let mut actual = String::new();
        write_structs(&mut actual, 0, &module);

        assert_eq!(
            indoc! {
                r"
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct VectorsF32 {
                    pub a: [f32; 2],
                    pub b: [f32; 3],
                    pub c: [f32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct VectorsU32 {
                    pub a: [u32; 2],
                    pub b: [u32; 3],
                    pub c: [u32; 4],
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct MatricesF32 {
                    pub a: glam::Mat4,
                }
                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
                pub struct StaticArrays {
                    pub a: [u32; 5],
                    pub b: [f32; 3],
                    pub c: [glam::Mat4; 512],
                }
                "
            },
            actual
        );
    }

    #[test]
    fn bind_group_layouts_descriptors_compute() {
        // The actual content of the structs doesn't matter.
        // We only care about the groups and bindings.
        let source = indoc! {r#"
            struct VertexInput0 {};
            struct VertexWeight {};
            struct Vertices {};
            struct VertexWeights {};
            struct Transforms {};

            [[group(0), binding(0)]] var<storage, read> src : Vertices;
            [[group(0), binding(1)]] var<storage, read> vertex_weights : VertexWeights;
            [[group(0), binding(2)]] var<storage, read_write> dst : Vertices;

            [[group(1), binding(0)]] var<uniform> transforms: Transforms;

            [[stage(compute)]]
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module);

        let mut actual = String::new();
        for (group_no, group) in bind_group_data {
            write_bind_group_layout(&mut actual, 0, group_no, &group);
            write_bind_group_layout_descriptor(
                &mut actual,
                0,
                group_no,
                &group,
                wgpu::ShaderStages::COMPUTE,
            );
        }

        assert_eq!(
            indoc! {
                r"
                pub struct BindGroupLayout0<'a> {
                    pub src: &'a wgpu::Buffer,
                    pub vertex_weights: &'a wgpu::Buffer,
                    pub dst: &'a wgpu::Buffer,
                }
                const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0u32,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1u32,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2u32,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ]
                };
                pub struct BindGroupLayout1<'a> {
                    pub transforms: &'a wgpu::Buffer,
                }
                const LAYOUT_DESCRIPTOR1: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0u32,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ]
                };
                "
            },
            actual
        );
    }

    #[test]
    fn bind_group_layouts_descriptors_vertex_fragment() {
        // The actual content of the structs doesn't matter.
        // We only care about the groups and bindings.
        let source = indoc! {r#"
            struct Transforms {};

            [[group(0), binding(0)]]
            var color_texture: texture_2d<f32>;
            [[group(0), binding(1)]]
            var color_sampler: sampler;
            [[group(1), binding(0)]] var<uniform> transforms: Transforms;

            [[stage(vertex)]]
            fn vs_main() {}

            [[stage(fragment)]]
            fn fs_main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let bind_group_data = wgsl::get_bind_group_data(&module);

        let mut actual = String::new();
        for (group_no, group) in bind_group_data {
            write_bind_group_layout(&mut actual, 0, group_no, &group);
            write_bind_group_layout_descriptor(
                &mut actual,
                0,
                group_no,
                &group,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
            );
        }

        // TODO: Are storage buffers valid for vertex/fragment?
        assert_eq!(
            indoc! {
                r"
                pub struct BindGroupLayout0<'a> {
                    pub color_texture: &'a wgpu::TextureView,
                    pub color_sampler: &'a wgpu::Sampler,
                }
                const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0u32,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1u32,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ]
                };
                pub struct BindGroupLayout1<'a> {
                    pub transforms: &'a wgpu::Buffer,
                }
                const LAYOUT_DESCRIPTOR1: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0u32,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ]
                };
                "
            },
            actual
        );
    }
}
