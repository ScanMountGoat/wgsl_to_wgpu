# wgsl_to_wgpu
[![Latest Version](https://img.shields.io/crates/v/wgsl_to_wgpu.svg)](https://crates.io/crates/wgsl_to_wgpu) [![docs.rs](https://docs.rs/wgsl_to_wgpu/badge.svg)](https://docs.rs/wgsl_to_wgpu)  
An experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).

wgsl_to_wgpu is designed to be incorporated into the compilation process using a build script. The WGSL shaders are parsed using [naga](https://github.com/gfx-rs/naga) to generate a corresponding Rust module. The generated Rust module contains the type definitions and boilerplate code needed to work with the WGSL shader module. Using the generated code can also reduce many instances of invalid API usage. See the docs.rs documentation for more detailed descriptions of the features and limitations of the generated code.

## Features
- bind groups and bindings
- shader module initialization
- Rust structs for vertex, storage, and uniform buffers
- derive encase and bytemuck
- const validation of WGSL memory layout for structs when using bytemuck

## Usage

The generated code currently relies on [memoffset](https://crates.io/crates/memoffset) for calculating field offsets for vertex input structs.
Add the following lines to the `Cargo.toml` and fill in the appropriate versions for `memoffset` and `wgsl_to_wgpu`.

```toml
[dependencies]
memoffset = "..."

[build-dependencies]
wgsl_to_wgpu = "..."
```

See the example crate for how to use the generated code. Run the example with `cargo run`.