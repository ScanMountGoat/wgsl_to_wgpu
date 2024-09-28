struct Transforms {
    value: vec4<f32>
};

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

@group(0) @binding(11)
var color_texture_array_2d: texture_2d_array<f32>;
@group(0) @binding(12)
var color_texture_array_cube: texture_cube_array<f32>;

@group(1) @binding(0) var<uniform> transforms: Transforms;
@group(1) @binding(1) var<uniform> scalar: f32;
@group(1) @binding(2) var<uniform> vector: vec4<f32>;
@group(1) @binding(3) var<uniform> matrix: mat4x4<f32>;

fn use_vars() {
    var x = transforms.value.x;
    x = scalar;
    x = vector.x;
    x = matrix[0][0];
    x = textureLoad(color_texture1, vec2(0), 0);
    x = textureLoad(color_texture2, vec2(0), 0);
    x = textureSample(color_texture3, color_sampler, vec2(0.0)).x;
    x = textureSampleCompare(depth_texture, comparison_sampler, vec2(0.0), 0.0);
    x = textureLoad(storage_tex_read, vec2(0));
    x = textureLoad(storage_tex_read_write, vec2(0));
    x = textureLoad(color_texture_array_2d, vec2(0), 0, 0);
    x = textureLoad(color_texture_array_cube, vec3(0), 0, 0);
}

@vertex
fn vs_main() {
    use_vars();
}

@fragment
fn fs_main() {
    use_vars();
}