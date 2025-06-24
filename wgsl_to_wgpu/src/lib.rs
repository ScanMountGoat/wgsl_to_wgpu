//! # wgsl_to_wgpu
//! wgsl_to_wgpu is a library for generating typesafe Rust bindings from WGSL shaders to [wgpu](https://github.com/gfx-rs/wgpu).
//!
//! ## Getting Started
//! The [create_shader_module] and [create_shader_modules] functions are intended for use in build scripts.
//! This facilitates a shader focused workflow where edits to WGSL code are automatically reflected in the corresponding Rust file.
//! For example, changing the type of a uniform in WGSL will raise a compile error in Rust code using the generated struct to initialize the buffer.
//!
//! ```rust no_run
//! // build.rs
//! use wgsl_to_wgpu::{create_shader_module, MatrixVectorTypes, WriteOptions};
//!
//! fn main() {
//!     println!("cargo:rerun-if-changed=src/shader.wgsl");
//!
//!     // Read the shader source file.
//!     let wgsl_source = std::fs::read_to_string("src/shader.wgsl").unwrap();
//!
//!     // Configure the output based on the dependencies for the project.
//!     let options = WriteOptions {
//!         derive_bytemuck_vertex: true,
//!         derive_encase_host_shareable: true,
//!         matrix_vector_types: MatrixVectorTypes::Glam,
//!         ..Default::default()
//!     };
//!
//!     // Generate the bindings.
//!     let text = create_shader_module(&wgsl_source, "shader.wgsl", options).unwrap();
//!     std::fs::write("src/shader.rs", text.as_bytes()).unwrap();
//! }
//! ```
//!
//! ## Modules and Preprocessors
//! There are a number of useful processing crates that extend or modify WGSL
//! to add features like module imports or preprocessor defines.
//! wgsl_to_wgpu does not provide support for any of these crates directly.
//! Instead, pass the final processed WGSL to [create_shader_modules]
//! and specify the approriate name demangling logic if needed.
//! See the function documentation for details.
#![allow(clippy::result_large_err)]

extern crate wgpu_types as wgpu;

use std::{
    collections::BTreeMap,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

use bindgroup::{bind_groups_module, get_bind_group_data};
use consts::pipeline_overridable_constants;
use entry::{entry_point_constants, fragment_states, vertex_states, vertex_struct_methods};
use naga::{valid::ValidationFlags, WithSpan};
use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;
use thiserror::Error;

mod bindgroup;
mod consts;
mod entry;
mod structs;
mod wgsl;

pub use naga::valid::Capabilities as WgslCapabilities;

/// Errors while generating Rust source for a WGSL shader module.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CreateModuleError {
    /// Bind group sets must be consecutive and start from 0.
    /// See `bind_group_layouts` for
    /// [PipelineLayoutDescriptor](https://docs.rs/wgpu/latest/wgpu/struct.PipelineLayoutDescriptor.html#).
    #[error("bind groups are non-consecutive or do not start from 0")]
    NonConsecutiveBindGroups,

    /// Each binding resource must be associated with exactly one binding index.
    #[error("duplicate binding found with index `{binding}`")]
    DuplicateBinding { binding: u32 },

    /// The shader source could not be parsed.
    #[error("failed to parse: {error}")]
    ParseError {
        error: naga::front::wgsl::ParseError,
    },

    /// The shader source could not be validated.
    #[error("failed to validate: {error}")]
    ValidationError {
        error: WithSpan<naga::valid::ValidationError>,
    },
}

impl CreateModuleError {
    /// Writes a diagnostic error to stderr.
    pub fn emit_to_stderr(&self, wgsl_source: &str) {
        match self {
            CreateModuleError::ParseError { error } => error.emit_to_stderr(wgsl_source),
            CreateModuleError::ValidationError { error } => error.emit_to_stderr(wgsl_source),
            other => {
                eprintln!("{other}")
            }
        }
    }

    /// Writes a diagnostic error to stderr, including a source path.
    pub fn emit_to_stderr_with_path(&self, wgsl_source: &str, path: impl AsRef<Path>) {
        let path = path.as_ref();
        match self {
            CreateModuleError::ParseError { error } => {
                error.emit_to_stderr_with_path(wgsl_source, path)
            }
            CreateModuleError::ValidationError { error } => {
                error.emit_to_stderr_with_path(wgsl_source, &path.to_string_lossy())
            }
            other => {
                eprintln!("{}: {}", path.to_string_lossy(), other)
            }
        }
    }

    /// Creates a diagnostic string from the error.
    pub fn emit_to_string(&self, wgsl_source: &str) -> String {
        match self {
            CreateModuleError::ParseError { error } => error.emit_to_string(wgsl_source),
            CreateModuleError::ValidationError { error } => error.emit_to_string(wgsl_source),
            other => {
                format!("{other}")
            }
        }
    }

    /// Creates a diagnostic string from the error, including a source path.
    pub fn emit_to_string_with_path(&self, wgsl_source: &str, path: impl AsRef<Path>) -> String {
        let path = path.as_ref();
        match self {
            CreateModuleError::ParseError { error } => {
                error.emit_to_string_with_path(wgsl_source, path)
            }
            CreateModuleError::ValidationError { error } => {
                error.emit_to_string_with_path(wgsl_source, &path.to_string_lossy())
            }
            other => {
                format!("{}: {}", path.to_string_lossy(), other)
            }
        }
    }
}

/// Options for configuring the generated bindings to work with additional dependencies.
/// Use [WriteOptions::default] for only requiring WGPU itself.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct WriteOptions {
    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for WGSL vertex input structs when `true`.
    pub derive_bytemuck_vertex: bool,

    /// Derive [bytemuck::Pod](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html#)
    /// and [bytemuck::Zeroable](https://docs.rs/bytemuck/latest/bytemuck/trait.Zeroable.html#)
    /// for user defined WGSL structs for host-shareable types (uniform and storage buffers) when `true`.
    ///
    /// This will generate compile time assertions to check that the memory layout
    /// of structs and struct fields matches what is expected by WGSL.
    /// This does not account for all layout and alignment rules like storage buffer offset alignment.
    ///
    /// Most applications should instead handle these requirements more reliably at runtime using encase.
    pub derive_bytemuck_host_shareable: bool,

    /// Derive [encase::ShaderType](https://docs.rs/encase/latest/encase/trait.ShaderType.html#)
    /// for user defined WGSL structs for host-shareable types (uniform and storage buffers) when `true`.
    /// Use [MatrixVectorTypes::Glam] for best compatibility.
    pub derive_encase_host_shareable: bool,

    /// Derive [serde::Serialize](https://docs.rs/serde/1.0.159/serde/trait.Serialize.html)
    /// and [serde::Deserialize](https://docs.rs/serde/1.0.159/serde/trait.Deserialize.html)
    /// for user defined WGSL structs when `true`.
    pub derive_serde: bool,

    /// The format to use for matrix and vector types.
    pub matrix_vector_types: MatrixVectorTypes,

    /// Format the generated code with the `rustfmt` formatter used for `cargo fmt`.
    /// This invokes a separate process to run the `rustfmt` executable.
    /// For cases where `rustfmt` is not available
    /// or the generated code is not included in the src directory,
    /// leave this at its default value of `false`.
    pub rustfmt: bool,

    /// Perform semantic validation on the code.
    pub validate: Option<ValidationOptions>,
}

/// Options for semantic validation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ValidationOptions {
    /// The IR capabilities to support.
    pub capabilities: WgslCapabilities,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            capabilities: WgslCapabilities::all(),
        }
    }
}

/// The format to use for matrix and vector types.
/// Note that the generated types for the same WGSL type may differ in size or alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixVectorTypes {
    /// Rust types like `[f32; 4]` or `[[f32; 4]; 4]`.
    Rust,

    /// `glam` types like `glam::Vec4` or `glam::Mat4`.
    /// Types not representable by `glam` like `mat2x3<f32>` will use the output from [MatrixVectorTypes::Rust].
    Glam,

    /// `nalgebra` types like `nalgebra::SVector<f64, 4>` or `nalgebra::SMatrix<f32, 2, 3>`.
    Nalgebra,
}

impl Default for MatrixVectorTypes {
    fn default() -> Self {
        Self::Rust
    }
}

/// Create a Rust module for a WGSL shader included via [include_str].
///
/// The `wgsl_include_path` should be a valid input to [include_str] in the generated file's location.
/// The included contents should be identical to `wgsl_source`.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/**
```rust no_run
// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wgsl_file = "src/shader.wgsl";
    println!("cargo:rerun-if-changed={wgsl_file}");

    // Read the shader source file.
    let wgsl_source = std::fs::read_to_string(wgsl_file)?;

    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions {
        derive_bytemuck_vertex: true,
        derive_encase_host_shareable: true,
        matrix_vector_types: wgsl_to_wgpu::MatrixVectorTypes::Glam,
        validate: Some(Default::default()),
        ..Default::default()
    };

    // Generate the bindings.
    let text = wgsl_to_wgpu::create_shader_module(&wgsl_source, "shader.wgsl", options)
        .inspect_err(|error| error.emit_to_stderr_with_path(&wgsl_source, wgsl_file))
        // Don't print out same error twice
        .map_err(|_| "Failed to validate shader")?;

    std::fs::write("src/shader.rs", text.as_bytes())?;
    Ok(())
}
```
 */
pub fn create_shader_module(
    wgsl_source: &str,
    wgsl_include_path: &str,
    options: WriteOptions,
) -> Result<String, CreateModuleError> {
    let mut root = Module::default();
    root.add_shader_module(
        wgsl_source,
        Some(wgsl_include_path),
        options,
        ModulePath::default(),
        demangle_identity,
    )?;
    Ok(root.to_generated_bindings(options))
}

/// Create Rust module(s) for a WGSL shader included as a string literal.
///
/// This creates a [Module] internally and adds a single WGSL shader.
/// See [Module::add_shader_module] for details.
///
/// # Examples
/// This function is intended to be called at build time such as in a build script.
/// The source string does not need to be from an actual file on disk.
/// This allows applying build time operations like preprocessor defines.
/**
```rust no_run
// build.rs
# fn generate_wgsl_source_string() -> String { String::new() }
# fn demangle(name: &str) -> wgsl_to_wgpu::TypePath { wgsl_to_wgpu::demangle_identity(name) }
use wgsl_to_wgpu::{create_shader_modules};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate the shader at build time.
    let wgsl_source = generate_wgsl_source_string();

    // Configure the output based on the dependencies for the project.
    let options = wgsl_to_wgpu::WriteOptions {
        derive_bytemuck_vertex: true,
        derive_encase_host_shareable: true,
        matrix_vector_types: wgsl_to_wgpu::MatrixVectorTypes::Glam,
        validate: Some(wgsl_to_wgpu::ValidationOptions {
            capabilities: wgsl_to_wgpu::WgslCapabilities::all(),
        }),
        ..Default::default()
    };

    // Generate the bindings.
    let text = create_shader_modules(&wgsl_source, options, demangle)
        .inspect_err(|error| error.emit_to_stderr(&wgsl_source))
        // Don't print out same error twice
        .map_err(|_| "Failed to validate shader")?;
    std::fs::write("src/shader.rs", text.as_bytes())?;
    Ok(())
}
```
 */
pub fn create_shader_modules<F>(
    wgsl_source: &str,
    options: WriteOptions,
    demangle: F,
) -> Result<String, CreateModuleError>
where
    F: Fn(&str) -> TypePath + Clone,
{
    let mut root = Module::default();
    root.add_shader_module(wgsl_source, None, options, ModulePath::default(), demangle)?;

    let output = root.to_generated_bindings(options);
    Ok(output)
}
/// A fully qualified absolute path like `a::b::Item` split into `["a", "b"]` and `Item`.
///
/// This path will be relative to the generated root module.
/// Use [ModulePath::default] to refer to the root module itself.
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct ModulePath {
    /// The path components like `["a", "b"]` in `a::b`.
    /// The root module has no components.
    pub components: Vec<String>,
}

/// A fully qualified absolute path like `a::b::Item` split into `["a", "b"]` and `Item`.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypePath {
    /// The parent components like `["a", "b"]` in `a::b::Item`.
    pub parent: ModulePath,
    /// The name of the item like `"Item"` for `a::b::Item`.
    pub name: String,
}

/// An identity demangling function that treats `name` as an item in the root module.
pub fn demangle_identity(name: &str) -> TypePath {
    TypePath {
        parent: ModulePath::default(),
        name: name.to_string(),
    }
}

/// Generated code for a Rust module and its submodules.
#[derive(Debug, Default)]
pub struct Module {
    items: BTreeMap<String, TokenStream>,
    submodules: BTreeMap<String, Module>,
}

impl Module {
    fn to_tokens(&self) -> TokenStream {
        let mut tokens = quote!();

        for item in self.items.values() {
            tokens = quote!(#tokens #item);
        }
        for (name, m) in &self.submodules {
            let submodule = m.to_tokens();

            let name = Ident::new(name, Span::call_site());
            tokens = quote! {
                #tokens
                pub mod #name {
                    #submodule
                }
            }
        }

        tokens
    }

    /// Generate a combined Rust module for this module and all of its submodules recursively.
    pub fn to_generated_bindings(&self, options: WriteOptions) -> String {
        let output = self.to_tokens();
        if options.rustfmt {
            pretty_print_rustfmt(output)
        } else {
            pretty_print(output)
        }
    }

    fn add_module_items(&mut self, structs: &[(TypePath, TokenStream)], root_path: &ModulePath) {
        for (item, tokens) in structs {
            // Replace a "root" path with the specified root module.
            let components = if item.parent.components.is_empty() {
                &root_path.components
            } else {
                &item.parent.components
            };
            let module = self.get_module(components);
            module.items.insert(item.name.clone(), tokens.clone());
        }
    }

    fn get_module<'a>(&'a mut self, parents: &[String]) -> &'a mut Module {
        if let Some((name, remaining)) = parents.split_first() {
            self.submodules
                .entry(name.clone())
                .or_default()
                .get_module(remaining)
        } else {
            self
        }
    }

    /// Add generated Rust code for a WGSL shader located at `root_path`.
    ///
    /// This should only be called for files with shader entry points.
    /// Imported shader files should be handled by the preprocessing library and included in `wgsl_source`.
    ///
    /// The `wgsl_include_path` should usually be `None` to include the `wgsl_source` as a string literal.
    /// If `Some`, the `wgsl_include_path` should be a valid input to [include_str] in the generated file's location
    /// with included contents identical to `wgsl_source`.
    ///
    /// # Name Demangling
    /// Name mangling is necessary in some cases to uniquely identify items and ensure valid WGSL names.
    /// The `demangle` function converts mangled absolute module paths to module path components.
    ///
    /// Use [demangle_identity] if the names do not need to be demangled.
    /// This demangle function will place all generated code in the root module.
    ///
    /// The demangling logic should reverse the operations performed by the mangling fuction.
    /// Some crates provide their own "demangle" or "undecorate" function as part of the public API.
    ///
    /// The demangle function used for wgsl_to_wgpu should demangle names into absolute module paths
    /// and split this absolute path into a parent [ModulePath] and item name.
    ///
    /// The [TypePath] uniquely identifies a generated item like a struct or constant.
    /// No guarantees are made about which item will be kept when generating multiple items
    /// with the same [TypePath].
    ///
    /// # Examples
    /**
    ```rust no_run
    // build.rs
    # fn generate_wgsl_source_string(_: &str) -> String { String::new() }
    # fn demangle(name: &str) -> wgsl_to_wgpu::TypePath { wgsl_to_wgpu::demangle_identity(name) }
    use wgsl_to_wgpu::{create_shader_modules, demangle_identity, Module, ModulePath};

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        // Configure the output based on the dependencies for the project.
        let options = wgsl_to_wgpu::WriteOptions::default();

        // Start with an empty root module.
        let mut root = Module::default();

        // Add a shader directly to the root module.
        let wgsl_source1 = generate_wgsl_source_string("src/root.wgsl");
        root.add_shader_module(&wgsl_source1, None, options, ModulePath::default(), demangle)?;

        // Add a shader in the "shader2" module.
        let wgsl_source2 = generate_wgsl_source_string("src/shader2.wgsl");
        let path2 = ModulePath {
            components: vec!["shader2".to_string()],
        };
        root.add_shader_module(&wgsl_source1, None, options, path2, demangle)?;

        // Generate modules for "shader1", "shader2", and any imported modules determined by the demangle function.
        let text = root.to_generated_bindings(options);

        std::fs::write("src/shaders.rs", text.as_bytes())?;
        Ok(())
    }
    ```
     */
    pub fn add_shader_module<F>(
        &mut self,
        wgsl_source: &str,
        wgsl_include_path: Option<&str>,
        options: WriteOptions,
        root_path: ModulePath,
        demangle: F,
    ) -> Result<(), CreateModuleError>
    where
        F: Fn(&str) -> TypePath + Clone,
    {
        let module = naga::front::wgsl::parse_str(wgsl_source)
            .map_err(|error| CreateModuleError::ParseError { error })?;

        if let Some(options) = options.validate.as_ref() {
            naga::valid::Validator::new(ValidationFlags::all(), options.capabilities)
                .validate(&module)
                .map_err(|error| CreateModuleError::ValidationError { error })?;
        }

        let global_stages = wgsl::global_shader_stages(&module);
        let bind_group_data = get_bind_group_data(&module, &global_stages, demangle.clone())?;

        let entry_stages = wgsl::entry_stages(&module);

        // Collect tokens for each item.
        let structs = structs::structs(&module, options, demangle.clone());
        let consts = consts::consts(&module, demangle.clone());
        let bind_groups_module = bind_groups_module(&module, &bind_group_data);
        let vertex_methods = vertex_struct_methods(&module, demangle.clone());
        let compute_module = compute_module(&module, demangle.clone());
        let entry_point_constants = entry_point_constants(&module, demangle.clone());
        let vertex_states = vertex_states(&module, demangle.clone());
        let fragment_states = fragment_states(&module, demangle.clone());

        // Use a string literal if no include path is provided.
        let included_source = wgsl_include_path
            .map(|p| quote!(include_str!(#p)))
            .unwrap_or_else(|| quote!(#wgsl_source));

        let create_shader_module = quote! {
            pub const SOURCE: &str = #included_source;
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                let source = std::borrow::Cow::Borrowed(SOURCE);
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(source)
                })
            }
        };

        let bind_group_layouts: Vec<_> = bind_group_data
            .keys()
            .map(|group_no| {
                let group = indexed_name_to_ident("BindGroup", *group_no);
                quote!(bind_groups::#group::get_bind_group_layout(device))
            })
            .collect();

        let (push_constant_range, push_constant_stages) =
            push_constant_range_stages(&module, &global_stages, entry_stages).unzip();

        let create_pipeline_layout = quote! {
            pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        #(&#bind_group_layouts),*
                    ],
                    push_constant_ranges: &[#push_constant_range],
                })
            }
        };

        let override_constants = pipeline_overridable_constants(&module, demangle);

        let push_constant_stages = push_constant_stages.map(|stages| {
            quote! {
                pub const PUSH_CONSTANT_STAGES: wgpu::ShaderStages = #stages;
            }
        });

        // Place items into appropriate modules.
        self.add_module_items(&consts, &root_path);
        self.add_module_items(&structs, &root_path);
        self.add_module_items(&vertex_methods, &root_path);

        // Place items generated for this module in the root module.
        let root_items = vec![(
            TypePath {
                parent: root_path.clone(),
                name: String::new(),
            },
            quote! {
                #override_constants
                #bind_groups_module
                #compute_module
                #entry_point_constants
                #vertex_states
                #fragment_states
                #create_shader_module
                #push_constant_stages
                #create_pipeline_layout
            },
        )];
        self.add_module_items(&root_items, &root_path);

        Ok(())
    }
}

fn push_constant_range_stages(
    module: &naga::Module,
    global_stages: &BTreeMap<String, wgpu::ShaderStages>,
    entry_stages: wgpu::ShaderStages,
) -> Option<(TokenStream, TokenStream)> {
    // Assume only one variable is used with var<push_constant> in WGSL.
    let (_, global) = module
        .global_variables
        .iter()
        .find(|(_, g)| g.space == naga::AddressSpace::PushConstant)?;

    let push_constant_size = module.types[global.ty].inner.size(module.to_ctx());

    // Set visibility to all stages that access this binding.
    // Use all entry points as a safe fallback.
    let shader_stages = global
        .name
        .as_ref()
        .and_then(|n| global_stages.get(n).copied())
        .unwrap_or(entry_stages);

    let stages = quote_shader_stages(shader_stages);

    // Use a single push constant range for all shader stages.
    // This allows easily setting push constants in a single call with offset 0.
    let size = Literal::usize_unsuffixed(push_constant_size as usize);
    Some((
        quote! {
            wgpu::PushConstantRange {
                stages: PUSH_CONSTANT_STAGES,
                range: 0..#size
            }
        },
        stages,
    ))
}

fn pretty_print(output: TokenStream) -> String {
    let file = syn::parse_file(&output.to_string()).unwrap();
    prettyplease::unparse(&file)
}

fn pretty_print_rustfmt(tokens: TokenStream) -> String {
    let value = tokens.to_string();
    // TODO: Return errors?
    if let Ok(mut proc) = Command::new("rustfmt")
        .arg("--emit=stdout")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        let stdin = proc.stdin.as_mut().unwrap();
        stdin.write_all(value.as_bytes()).unwrap();

        let output = proc.wait_with_output().unwrap();
        if output.status.success() {
            // Don't modify line endings.
            return String::from_utf8(output.stdout).unwrap().replace("\r", "");
        }
    }
    value.to_string()
}

fn indexed_name_to_ident(name: &str, index: u32) -> Ident {
    Ident::new(&format!("{name}{index}"), Span::call_site())
}

fn compute_module<F>(module: &naga::Module, demangle: F) -> TokenStream
where
    F: Fn(&str) -> TypePath + Clone,
{
    let entry_points: Vec<_> = module
        .entry_points
        .iter()
        .filter_map(|e| {
            if e.stage == naga::ShaderStage::Compute {
                let workgroup_size_constant = workgroup_size(e, demangle.clone());
                let create_pipeline = create_compute_pipeline(e, demangle.clone());

                Some(quote! {
                    #workgroup_size_constant
                    #create_pipeline
                })
            } else {
                None
            }
        })
        .collect();

    if entry_points.is_empty() {
        // Don't include empty modules.
        quote!()
    } else {
        quote! {
            pub mod compute {
                #(#entry_points)*
            }
        }
    }
}

fn create_compute_pipeline<F>(e: &naga::EntryPoint, demangle: F) -> TokenStream
where
    F: Fn(&str) -> TypePath,
{
    let name = &demangle(&e.name).name;

    // Compute pipeline creation has few parameters and can be generated.
    let pipeline_name = Ident::new(&format!("create_{}_pipeline", name), Span::call_site());

    // The entry name string itself should remain mangled to match the WGSL code.
    let entry_point = &e.name;

    // TODO: Include a user supplied module name in the label?
    let label = format!("Compute Pipeline {}", name);
    quote! {
        pub fn #pipeline_name(device: &wgpu::Device) -> wgpu::ComputePipeline {
            let module = super::create_shader_module(device);
            let layout = super::create_pipeline_layout(device);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(#label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(#entry_point),
                compilation_options: Default::default(),
                cache: Default::default(),
            })
        }
    }
}

fn workgroup_size<F>(e: &naga::EntryPoint, demangle: F) -> TokenStream
where
    F: Fn(&str) -> TypePath + Clone,
{
    let name = &demangle(&e.name).name;

    let name = Ident::new(
        &format!("{}_WORKGROUP_SIZE", name.to_uppercase()),
        Span::call_site(),
    );
    let [x, y, z] = e
        .workgroup_size
        .map(|s| Literal::usize_unsuffixed(s as usize));
    quote!(pub const #name: [u32; 3] = [#x, #y, #z];)
}

fn quote_shader_stages(stages: wgpu::ShaderStages) -> TokenStream {
    if stages == wgpu::ShaderStages::all() {
        quote!(wgpu::ShaderStages::all())
    } else if stages == wgpu::ShaderStages::VERTEX_FRAGMENT {
        quote!(wgpu::ShaderStages::VERTEX_FRAGMENT)
    } else {
        let mut components = Vec::new();
        if stages.contains(wgpu::ShaderStages::VERTEX) {
            components.push(quote!(wgpu::ShaderStages::VERTEX));
        }
        if stages.contains(wgpu::ShaderStages::FRAGMENT) {
            components.push(quote!(wgpu::ShaderStages::FRAGMENT));
        }
        if stages.contains(wgpu::ShaderStages::COMPUTE) {
            components.push(quote!(wgpu::ShaderStages::COMPUTE));
        }

        if let Some((first, remaining)) = components.split_first() {
            quote!(#first #(.union(#remaining))*)
        } else {
            quote!(wgpu::ShaderStages::NONE)
        }
    }
}

// Tokenstreams can't be compared directly using PartialEq.
// Use pretty_print to normalize the formatting and compare strings.
// Use a colored diff output to make differences easier to see.
#[cfg(test)]
#[macro_export]
macro_rules! assert_tokens_eq {
    ($a:expr, $b:expr) => {
        pretty_assertions::assert_eq!(
            crate::pretty_print_rustfmt($a),
            crate::pretty_print_rustfmt($b)
        )
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[test]
    fn create_shader_module_include_source() {
        let source = indoc! {r#"
            var<push_constant> consts: vec4<f32>;

            @fragment
            fn fs_main() {}
        "#};

        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default())
            .unwrap()
            .parse()
            .unwrap();

        assert_tokens_eq!(
            quote! {
                pub const ENTRY_FS_MAIN: &str = "fs_main";
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
                pub fn fs_main_entry(targets: [Option<wgpu::ColorTargetState>; 0]) -> FragmentEntry<0> {
                    FragmentEntry {
                        entry_point: ENTRY_FS_MAIN,
                        targets,
                        constants: Default::default(),
                    }
                }
                pub const SOURCE: &str = include_str!("shader.wgsl");
                pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                    let source = std::borrow::Cow::Borrowed(SOURCE);
                    device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Wgsl(source),
                        })
                }
                pub const PUSH_CONSTANT_STAGES: wgpu::ShaderStages = wgpu::ShaderStages::FRAGMENT;
                pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
                    device
                        .create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[],
                                push_constant_ranges: &[
                                    wgpu::PushConstantRange {
                                        stages: PUSH_CONSTANT_STAGES,
                                        range: 0..16,
                                    },
                                ],
                            },
                        )
                }
            },
            actual
        );
    }

    #[test]
    fn create_shader_modules_source() {
        let source = include_str!("data/fragment_simple.wgsl");
        let actual =
            create_shader_modules(source, WriteOptions::default(), demangle_identity).unwrap();
        assert_eq!(include_str!("data/fragment_simple.rs"), actual);
    }

    #[test]
    fn create_shader_modules_source_rustfmt() {
        let source = include_str!("data/fragment_simple.wgsl");
        let actual = create_shader_modules(
            source,
            WriteOptions {
                rustfmt: true,
                ..Default::default()
            },
            demangle_identity,
        )
        .unwrap();
        assert_eq!(include_str!("data/fragment_simple_rustfmt.rs"), actual);
    }

    #[test]
    fn create_shader_module_consecutive_bind_groups() {
        let source = indoc! {r#"
            struct A {
                f: vec4<f32>
            };
            @group(0) @binding(0) var<uniform> a: A;
            @group(1) @binding(0) var<uniform> b: f32;
            @group(2) @binding(0) var<uniform> c: vec4<f32>;
            @group(3) @binding(0) var<uniform> d: mat4x4<f32>;

            @vertex
            fn vs_main() {}

            @fragment
            fn fs_main() {}
        "#};

        create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
    }

    #[test]
    fn create_shader_module_non_consecutive_bind_groups() {
        let source = indoc! {r#"
            @group(0) @binding(0) var<uniform> a: vec4<f32>;
            @group(1) @binding(0) var<uniform> b: vec4<f32>;
            @group(3) @binding(0) var<uniform> c: vec4<f32>;

            @fragment
            fn main() {}
        "#};

        let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());
        assert!(matches!(
            result,
            Err(CreateModuleError::NonConsecutiveBindGroups)
        ));
    }

    #[test]
    fn create_shader_module_repeated_bindings() {
        let source = indoc! {r#"
            struct A {
                f: vec4<f32>
            };
            @group(0) @binding(2) var<uniform> a: A;
            @group(0) @binding(2) var<uniform> b: A;

            @fragment
            fn main() {}
        "#};

        let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());
        assert!(matches!(
            result,
            Err(CreateModuleError::DuplicateBinding { binding: 2 })
        ));
    }

    fn items_to_tokens(items: Vec<(TypePath, TokenStream)>) -> TokenStream {
        let mut root = Module::default();
        root.add_module_items(&items, &ModulePath::default());
        root.to_tokens()
    }

    #[test]
    fn write_vertex_module_empty() {
        let source = indoc! {r#"
            @vertex
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module, demangle_identity);

        assert_tokens_eq!(quote!(), items_to_tokens(actual));
    }

    #[test]
    fn write_vertex_module_single_input_float32() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: f32,
                @location(1) b: vec2<f32>,
                @location(2) c: vec3<f32>,
                @location(3) d: vec4<f32>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module, demangle_identity);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            items_to_tokens(actual)
        );
    }

    #[test]
    fn write_vertex_module_single_input_float64() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: f64,
                @location(1) b: vec2<f64>,
                @location(2) c: vec3<f64>,
                @location(3) d: vec4<f64>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module, demangle_identity);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float64x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            items_to_tokens(actual)
        );
    }

    #[test]
    fn write_vertex_module_single_input_float16() {
        let source = indoc! {r#"
            enable f16;

            struct VertexInput0 {
                @location(0) a: f16,
                @location(1) b: vec2<f16>,
                @location(2) c: vec4<f16>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module, demangle_identity);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float16,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float16x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float16x4,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            items_to_tokens(actual)
        );
    }

    #[test]
    fn write_vertex_module_single_input_sint32() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: i32,
                @location(1) b: vec2<i32>,
                @location(2) c: vec3<i32>,
                @location(3) d: vec4<i32>,

            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module, demangle_identity);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Sint32x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            items_to_tokens(actual)
        );
    }

    #[test]
    fn write_vertex_module_single_input_uint32() {
        let source = indoc! {r#"
            struct VertexInput0 {
                @location(0) a: u32,
                @location(1) b: vec2<u32>,
                @location(2) c: vec3<u32>,
                @location(3) d: vec4<u32>,
            };

            @vertex
            fn main(in0: VertexInput0) {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = vertex_struct_methods(&module, demangle_identity);

        assert_tokens_eq!(
            quote! {
                impl VertexInput0 {
                    pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: std::mem::offset_of!(VertexInput0, a) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x2,
                            offset: std::mem::offset_of!(VertexInput0, b) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x3,
                            offset: std::mem::offset_of!(VertexInput0, c) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x4,
                            offset: std::mem::offset_of!(VertexInput0, d) as u64,
                            shader_location: 3,
                        },
                    ];
                    pub const fn vertex_buffer_layout(
                        step_mode: wgpu::VertexStepMode,
                    ) -> wgpu::VertexBufferLayout<'static> {
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<VertexInput0>() as u64,
                            step_mode,
                            attributes: &VertexInput0::VERTEX_ATTRIBUTES,
                        }
                    }
                }
            },
            items_to_tokens(actual)
        );
    }

    #[test]
    fn write_compute_module_empty() {
        let source = indoc! {r#"
            @vertex
            fn main() {}
        "#};

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = compute_module(&module, demangle_identity);

        assert_tokens_eq!(quote!(), actual);
    }

    #[test]
    fn write_compute_module_multiple_entries() {
        let source = indoc! {r#"
            @compute
            @workgroup_size(1,2,3)
            fn main1() {}

            @compute
            @workgroup_size(256)
            fn main2() {}
        "#
        };

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let actual = compute_module(&module, demangle_identity);

        assert_tokens_eq!(
            quote! {
                pub mod compute {
                    pub const MAIN1_WORKGROUP_SIZE: [u32; 3] = [1, 2, 3];
                    pub fn create_main1_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
                        let module = super::create_shader_module(device);
                        let layout = super::create_pipeline_layout(device);
                        device
                            .create_compute_pipeline(
                                &wgpu::ComputePipelineDescriptor {
                                    label: Some("Compute Pipeline main1"),
                                    layout: Some(&layout),
                                    module: &module,
                                    entry_point: Some("main1"),
                                    compilation_options: Default::default(),
                                    cache: Default::default(),
                                },
                            )
                    }
                    pub const MAIN2_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
                    pub fn create_main2_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
                        let module = super::create_shader_module(device);
                        let layout = super::create_pipeline_layout(device);
                        device
                            .create_compute_pipeline(
                                &wgpu::ComputePipelineDescriptor {
                                    label: Some("Compute Pipeline main2"),
                                    layout: Some(&layout),
                                    module: &module,
                                    entry_point: Some("main2"),
                                    compilation_options: Default::default(),
                                    cache: Default::default(),
                                },
                            )
                    }
                }
            },
            actual
        );
    }

    #[test]
    fn quote_all_shader_stages() {
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::NONE),
            quote_shader_stages(wgpu::ShaderStages::NONE)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::VERTEX),
            quote_shader_stages(wgpu::ShaderStages::VERTEX)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::FRAGMENT),
            quote_shader_stages(wgpu::ShaderStages::FRAGMENT)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::COMPUTE),
            quote_shader_stages(wgpu::ShaderStages::COMPUTE)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::VERTEX_FRAGMENT),
            quote_shader_stages(wgpu::ShaderStages::VERTEX_FRAGMENT)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::COMPUTE)),
            quote_shader_stages(wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::FRAGMENT.union(wgpu::ShaderStages::COMPUTE)),
            quote_shader_stages(wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE)
        );
        assert_tokens_eq!(
            quote!(wgpu::ShaderStages::all()),
            quote_shader_stages(wgpu::ShaderStages::all())
        );
    }

    #[test]
    fn create_shader_module_parse_error() {
        let source = indoc! {r#"
            var<push_constant> consts: vec4<f32>;

            @fragment
            fn fs_main() }
        "#};

        let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());

        assert!(
            matches!(result, Err(CreateModuleError::ParseError { .. })),
            "{result:?} is ParseError"
        )
    }

    #[test]
    fn create_shader_module_semantic_error() {
        let source = indoc! {r#"
            var<push_constant> consts: vec4<f32>;

            @fragment
            fn fs_main() {
                consts.x = 1;
            }
        "#};

        let result = create_shader_module(
            source,
            "shader.wgsl",
            WriteOptions {
                validate: Some(Default::default()),
                ..Default::default()
            },
        );

        assert!(
            matches!(result, Err(CreateModuleError::ValidationError { .. }),),
            "{result:?} is ValidationError"
        )
    }
}
