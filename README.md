# wgsl_to_wgpu
[![Latest Version](https://img.shields.io/crates/v/wgsl_to_wgpu.svg)](https://crates.io/crates/wgsl_to_wgpu) [![docs.rs](https://docs.rs/wgsl_to_wgpu/badge.svg)](https://docs.rs/wgsl_to_wgpu)  
An experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).

## Usage
The WGSL shaders are parsed using [naga](https://github.com/gfx-rs/naga) to generate corresponding types in Rust.
The provided functions can be incorporated into the compilation process using a build script.
This enables catching many instances of invalid API usage at compile time such as incorrectly configuring group and binding indices.
The amount of boilerplate code needed to initialize data in WGSL shaders is greatly reduced since binding layouts and descriptor code is generated automatically.

See the example crate for how to use the generated code. Run the example with `cargo run`.

## Limitations
This project supports most WGSL types but doesn't enforce certain key properties such as field alignment.
It may be necessary to disable running this function for shaders with unsupported types or features. The current implementation assumes all shader stages are part of a single WGSL source file. Vertex attributes using floating point types in WGSL like `vec2<f32>` are assumed to use float inputs
instead of normalized attributes like unorm or snorm integers. Insufficient or innaccurate generated code should be replaced by handwritten implementations as needed.

## Credits
- [naga](https://github.com/gfx-rs/naga) - WGSL parser and syntax
- [wgpu](https://github.com/gfx-rs/wgpu) - Rust implementation of WebGPU