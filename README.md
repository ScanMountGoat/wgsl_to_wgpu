# wgsl_to_wgpu
An experimental library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu) using macros. 
The WGSL shaders are parsed at compile time using [naga](https://github.com/gfx-rs/naga) to generate corresponding types in Rust. 
This enables catching many instances of invalid API usage at compile time such as incorrectly configuring group and binding indices.
The amount of boilerplate code needed to initialize data in WGSL shaders is greatly reduced since binding layouts and descriptor code is generated automatically.
The library currently only has limited support for bind groups.

# Credits
- [naga](https://github.com/gfx-rs/naga) - WGSL parser and syntax
- [wgpu](https://github.com/gfx-rs/wgpu) - Rust implementation of WebGPU