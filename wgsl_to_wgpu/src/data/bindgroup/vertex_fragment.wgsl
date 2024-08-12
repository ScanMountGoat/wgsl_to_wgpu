struct Transforms {};

@group(0) @binding(0)
var color_texture1: texture_2d<u32>;
@group(0) @binding(1)
var color_texture2: texture_2d<i32>;
@group(0) @binding(2)
var color_texture3: texture_2d<f32>;
@group(0) @binding(3)
var depth_texture: texture_depth_2d;

@group(0) @binding(4)
var color_sampler: sampler;
@group(0) @binding(5)
var comparison_sampler: sampler_comparison;

@group(0) @binding(6)
var storage_tex_read: texture_storage_2d<r32float, read>;
@group(0) @binding(7)
var storage_tex_write: texture_storage_2d<rg32sint, write>;
@group(0) @binding(8)
var storage_tex_read_write: texture_storage_2d<rgba8uint, read_write>;

@group(0) @binding(9)
var color_texture_msaa: texture_multisampled_2d<f32>;
@group(0) @binding(10)
var depth_texture_msaa: texture_depth_multisampled_2d;

@group(1) @binding(0) var<uniform> transforms: Transforms;

@vertex
fn vs_main() {}

@fragment
fn fs_main() {}