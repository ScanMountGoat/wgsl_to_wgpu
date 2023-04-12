# wgsl_to_wgpu
[![Latest Version](https://img.shields.io/crates/v/wgsl_to_wgpu.svg)](https://crates.io/crates/wgsl_to_wgpu) [![docs.rs](https://docs.rs/wgsl_to_wgpu/badge.svg)](https://docs.rs/wgsl_to_wgpu)  
An experimental library for generating typesafe Rust bindings from [WGSL](https://www.w3.org/TR/WGSL/) shaders to [wgpu](https://github.com/gfx-rs/wgpu).

wgsl_to_wgpu is designed to be incorporated into the compilation process using a build script. The WGSL shaders are parsed using [naga](https://github.com/gfx-rs/naga) to generate a corresponding Rust module. The generated Rust module contains the type definitions and boilerplate code needed to work with the WGSL shader module. Using the generated code can also reduce many instances of invalid API usage. wgsl_to_wgpu facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file. For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.

## Features
- more strongly typed bind group and bindings initialization
- shader module initialization
- Rust structs for vertex, storage, and uniform buffers
- optional derives for encase, bytemuck, and serde
- const validation of WGSL memory layout for generated structs when using bytemuck

## Usage
The generated code currently relies on [memoffset](https://crates.io/crates/memoffset) for calculating field offsets for vertex input structs.
Add the following lines to the `Cargo.toml` and fill in the appropriate versions for `memoffset` and `wgsl_to_wgpu`.
When enabling derives for crates like bytemuck, serde, or encase, these dependencies should also be added to the `Cargo.toml` with the appropriate derive features.

```toml
[dependencies]
memoffset = "..."

[build-dependencies]
wgsl_to_wgpu = "..."
```

See the example crate for how to use the generated code. Run the example with `cargo run`.

## Limitations
- It may be necessary to disable running this function for shaders with unsupported types or features.
Please make an issue if any new or existing WGSL syntax is unsupported.
- This library is not a rendering library and will not generate any high level abstractions like a material or scene graph. 
The goal is just to generate most of the tedious and error prone boilerplate required to use WGSL shaders with wgpu.
- The generated code will not prevent accidentally calling a function from an unrelated generated module.
It's recommended to name the shader module with the same name as the shader and use unique shader names to avoid issues. 
Using generated code from a different shader module may be desirable in some cases such as using the same camera struct definition in multiple WGSL shaders.
- The current implementation assumes all shader stages are part of a single WGSL source file. Shader modules split across files may be supported in a future release.
- Uniform and storage buffers can be initialized using the wrong generated Rust struct. 
WGPU will still validate the size of the buffer binding at runtime.
- Most but not all WGSL types are currently supported.
- Vertex attributes using floating point types in WGSL like `vec2<f32>` are assumed to use float inputs instead of normalized attributes like unorm or snorm integers.
- It's possible to achieve slightly better performance than the generated code in some cases like avoiding redundant bind group bindings or adjusting resource shader stage visibility. 
This can be fixed by calling lower level generated functions or handwriting functions as needed.