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
When enabling derives for crates like bytemuck, serde, or encase, these dependencies should also be added to the `Cargo.toml` with the appropriate derive features. See the provided [example project](https://github.com/ScanMountGoat/wgsl_to_wgpu/tree/main/example) for basic usage.

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
- All textures are assumed to be filterable and all samplers are assumed to be filtering. This may lead to compatibility issues. This can usually be resolved by requesting the native only feature TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES.
- It's possible to achieve slightly better performance than the generated code in some cases like avoiding redundant bind group bindings or adjusting resource shader stage visibility. This should be addressed by using some handwritten code where appropriate. See [descriptor table frequency (DX12)](https://learn.microsoft.com/en-us/windows/win32/direct3d12/advanced-use-of-descriptor-tables#changing-descriptor-table-entries-between-rendering-calls) and [descriptor set frequency (Vulkan)](https://vkguide.dev/docs/chapter-4/descriptors/#mental-model). 

## Publishing Crates
Rust expects build scripts to not modify files outside of OUT_DIR. The provided example project outputs the generated bindings to the `src/` directory for documentation purposes. 
This approach is also fine for applications, but published packages should follow the recommendations for build scripts in the [Cargo Book](https://doc.rust-lang.org/cargo/reference/build-scripts.html#case-study-code-generation).

```rust
use wgsl_to_wgpu::{create_shader_module_embedded, WriteOptions};

// src/build.rs
fn main() {
    println!("cargo:rerun-if-changed=src/model.wgsl");

    // Generate the Rust bindings and write to a file.
    let text = create_shader_module_embedded(wgsl_source, WriteOptions::default()).unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/model.rs"), text.as_bytes()).unwrap();
}
```

The generated code will need to be included in one of the normal source files. This includes adding any nested modules as needed.

```rust
// src/shader.rs
pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model.rs"));
}
```