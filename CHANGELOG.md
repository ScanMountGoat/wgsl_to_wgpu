# Changelog

All notable changes to this project will be documented in this file.
Breaking changes in the generated code will be considered as breaking changes when versioning wgsl_to_wgpu itself.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.10.0 - 2024-08-29
### Added
* Added a `rustfmt` option for formatting the output code with the executable used for `cargo fmt`.

### Changed
* Changed generated bind groups and bind group layouts to use labels with the group index like "BindGroup0" or "BindGroupLayout0".
* Changed bind group set methods to be generic over the pass type to better support mixed entry function stages.

### Fixed
* Fixed an issue where vertex entry code would only be generated for the first vertex entry.
* Fixed an issue where `wgpu::ShaderStages` stages would not be correctly generated for modules with both compute and vertex or fragment entries.

## 0.9.0 - 2024-07-27
### Fixed
* Fixed an issue where push constants would not work on Metal due to setting unused shader stages.

## 0.8.1 - 2024-06-29
### Added
* Added support for a single push constant range used by all shader stages.

### Fixed
* Fixed an issue where texture sample types of `u32` and `i32` did not generate the correct `wgpu::TextureSampleType`.

## 0.8.0 - 2024-05-09
### Added
* Added support for pipeline-overrideable constants.
* Added `fragment_state` and fragment entry functions for more easily initializing `wgpu::FragmentState`.

### Changed
* Changed vertex entry functions to take `&OverrideConstants` if needed to intialize overrides.

## 0.7.0 - 2024-04-26
### Changed
* Split derives for bytemuck into `derive_bytemuck_vertex` and `derive_bytemuck_host_shareable` to `WriteOptions`.
* Renamed `derive_encase` to `derive_encase_host_shareable` for `WriteOptions` and adjusted encase derive to not apply to vertex input structs to avoid memory layout mismatches.
* Changed vertex input field offsets to use `std::mem::offset_of!` instead of `memoffset::offset_of!`. This requires Rust 1.77 or later.

## 0.6.0 - 2024-02-23
### Added
* Added `BindGroups::set` method for setting all bind groups from a struct.

### Changed
* Moved vertex input methods from vertex module to top level.
* Moved `set_bind_groups` to top level and changed parameters to directly take bind group references to simplify code.

### Fixed
* Fixed an issue where `@builtin()` attributes were not supported in struct definitions.

### 0.5.0 - 2023-10-28
### Added
* Added `create_shader_module_embedded` for including the source as a string literal instead of using `include_str!`.
* Added `#[derive(Debug)]` to generated types for easier debugging and profiling.

## 0.4.1 - 2023-05-04
### Changed
* Adjusted code generation to skip structs that are only used internally in the shader.

### Fixed
* Update naga to 0.12.0 to match wgpu 0.16.

## 0.4.0 - 2023-04-30
### Added
* Added vertex buffer layout function to each vertex input struct.
* Added support for nalgebra for matrix and vector types.
* Added const asserts to check approriate Rust struct memory layouts with WGSL when deriving bytemuck.
* Added support for storage textures.
* Added support for multisampled textures.
* Added a function for creating compute pipelines.
* Added optional derives for `serde::Serialize` and `serde::Deserialize`.
* Added functions to initialize vertex state for pipeline descriptors.

### Changed
* Skip generating structs for shader stage outputs since they aren't needed.

## 0.3.1 - 2022-12-18
### Changed
* Updated documentation.

## 0.3.0 - 2022-12-13
### Added
* Added a check to force bind groups to be consecutive.
* Added a check for repeated bind groups or bindings.
* Added an example project.
* Added support for array bindings.
* Added support for depth textures.
* Added support for additional WGSL scalar types.
* Added `WriteOptions` to control generated derives.
* Added support for `glam` or Rust types for vectors and matrices.
* Added the workgroup size for generated code for compute shaders.

### Changed
* Changed the return type of `create_shader_module` to `String` instead of writing to a file.
* Changed the visibility of resources to match the shader stages present in the module.
* Changed shader source parameter of `create_shader_module` take `&str` instead of a path.
* Changed bind groups to be more flexible by taking `wgpu::BufferBinding` instead of `wgpu::Buffer`.
* Changed project to depend on `wgpu-types` instead of `wgpu` itself.

### Removed
* Removed the notice for generated code from the generated module string.
* Removed vertex attribute location code.
* Removed inaccurate generated struct size code.

## 0.2.0 - 2022-03-09
### Added
* Added create_pipeline_layout function.
* Added create_shader_module function.
* Added constants for vertex input locations.
* Added functions to set all bind groups or individual bind groups.
* Added Rust structs for global types like uniforms and vertex inputs.
* Added code to generate vertex state from vertex input structs.
* Added `#[derive(PartialEq)]` to generated structs.

### Fixed
* Fixed an issue where the generated code would not use `wgpu::ComputePass` for compute only modules.

### Changed
* Moved bindgroup layout descriptors to constants.
* Converted the procedural macro to a function to be used in build scripts.
* Always generate Rust types like arrays instead of forcing glam.

## 0.1.0 - 2021-07-10
Initial release!