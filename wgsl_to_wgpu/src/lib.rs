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
//! to add features like module imports or conditional compilation.
//! wgsl_to_wgpu does not provide support for any of these crates directly.
//! Instead, pass the final processed WGSL to [create_shader_modules]
//! and specify the approriate name demangling logic if needed.
//! See the function documentation for details.
#![allow(clippy::result_large_err)]

extern crate wgpu_types as wgpu;

use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

use bindgroup::{bind_group_modules, get_bind_group_data, set_bind_groups_func};
use consts::pipeline_overridable_constants;
use entry::{
    entry_point_constants, fragment_states, vertex_states, vertex_states_shared,
    vertex_struct_methods,
};
use naga::valid::ValidationFlags;
use proc_macro2::{Literal, Span, TokenStream};
use quote::{TokenStreamExt, quote};
use syn::Ident;

mod bindgroup;
mod compute;
mod consts;
mod entry;
mod error;
mod structs;
mod wgsl;

pub use error::CreateModuleError;
pub use naga::valid::Capabilities as WgslCapabilities;

use crate::{
    bindgroup::{GroupName, set_bind_groups_trait},
    compute::compute_module,
};

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

    /// Derive [serde::Serialize](https://docs.rs/serde/1.0.*/serde/trait.Serialize.html)
    /// and [serde::Deserialize](https://docs.rs/serde/1.0.*/serde/trait.Deserialize.html)
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

    /// Todo: Documentation
    pub named_bind_groups: bool,

    /// Todo: Documentation
    pub shared_bind_groups: bool,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixVectorTypes {
    /// Rust types like `[f32; 4]` or `[[f32; 4]; 4]`.
    #[default]
    Rust,

    /// `glam` types like `glam::Vec4` or `glam::Mat4`.
    /// Types not representable by `glam` like `mat2x3<f32>` will use the output from [MatrixVectorTypes::Rust].
    Glam,

    /// `nalgebra` types like `nalgebra::SVector<f64, 4>` or `nalgebra::SMatrix<f32, 2, 3>`.
    Nalgebra,
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
/// A fully qualified absolute module path like `a::b` split into `["a", "b"]`.
///
/// This path will be relative to the generated root module.
/// Use [ModulePath::default] to refer to the root module itself.
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct ModulePath {
    /// The path components like `["a", "b"]` in `a::b`.
    /// The root module has no components.
    pub components: Vec<String>,
}

impl ModulePath {
    fn relative_path(&self, target: &TypePath) -> TokenStream {
        self.relative_path_impl(&target.parent, Some(&target.name))
    }

    fn relative_module_path(&self, target: &ModulePath) -> TokenStream {
        self.relative_path_impl(target, None)
    }

    fn relative_path_impl(&self, module: &ModulePath, name: Option<&str>) -> TokenStream {
        let common = self.common_length(module);

        let to_common = self.components[common..].iter().map(|_| "super");
        let from_common = module.components[common..].iter();

        let base_path: PathBuf = self.components.iter().collect();
        let target_path: PathBuf = module.components.iter().collect();
        let relative_path = pathdiff::diff_paths(target_path, base_path).unwrap();

        // TODO: Implement this from scratch with tests?
        let components = relative_path
            .components()
            .filter_map(|c| match c {
                std::path::Component::Prefix(_) => None,
                std::path::Component::RootDir => None,
                std::path::Component::CurDir => None,
                std::path::Component::ParentDir => Some("super"),
                std::path::Component::Normal(s) => s.to_str(),
            })
            .chain(name)
            .map(|c| Ident::new(c, Span::call_site()));

        let mut path = TokenStream::new();
        path.append_separated(components, quote! { :: });
        path
    }

    fn common_prefix(&mut self, other: &ModulePath) {
        let common = self.common_length(other);
        self.components.truncate(common);
    }

    fn common_length(&self, other: &ModulePath) -> usize {
        self.components
            .iter()
            .zip(other.components.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(self.components.len())
            .min(other.components.len())
    }
}

/// A fully qualified absolute path like `a::b::Item` split into `["a", "b"]` and `"Item"`.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypePath {
    /// The parent components like `["a", "b"]` in `a::b::Item`.
    pub parent: ModulePath,
    /// The unique name of the item like `"Item"` for `a::b::Item`.
    ///
    /// This should be unique for all distinct items in a module.
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

    fn add_module_item(&mut self, path: TypePath, tokens: TokenStream) {
        let module = self.get_module(&path.parent.components);
        module.items.insert(path.name, tokens);
    }

    fn add_module_items(&mut self, items: Vec<(TypePath, TokenStream)>) {
        for (item, tokens) in items {
            self.add_module_item(item, tokens);
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

        // Generate modules for "shader1", "shader2",
        // and any imported modules determined by the demangle function.
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
        let demangle = demangle_with_root(demangle, root_path.clone());
        let module = naga::front::wgsl::parse_str(wgsl_source)
            .map_err(|error| CreateModuleError::ParseError { error })?;

        if let Some(options) = options.validate.as_ref() {
            naga::valid::Validator::new(ValidationFlags::all(), options.capabilities)
                .validate(&module)
                .map_err(|error| CreateModuleError::ValidationError { error })?;
        }

        let global_stages = wgsl::global_shader_stages(&module);
        let bind_group_data = get_bind_group_data(
            &module,
            &global_stages,
            demangle.clone(),
            options.named_bind_groups,
            options.shared_bind_groups,
        )?;

        // Collect tokens for each item.
        let structs = structs::structs(&module, options, demangle.clone());
        let consts = consts::consts(&module, demangle.clone());
        let vertex_methods = vertex_struct_methods(&module, demangle.clone());
        let entry_point_constants = entry_point_constants(&module, demangle.clone());
        let vertex_states = vertex_states(&module, demangle.clone());
        let bind_group_modules = bind_group_modules(&module, &bind_group_data, &root_path);

        let (shared_path, shared_items) = shared_root_module(&bind_group_data, &vertex_states);

        let (root_item_path, root_items) = shader_root_module(
            &module,
            wgsl_source,
            wgsl_include_path,
            root_path,
            &bind_group_data,
            demangle,
            bind_group_modules.root,
        );

        // Place items into appropriate modules.
        self.add_module_item(shared_path, shared_items);
        self.add_module_items(consts);
        self.add_module_items(structs);
        self.add_module_items(vertex_methods);
        self.add_module_items(entry_point_constants);
        self.add_module_items(vertex_states);
        self.add_module_items(bind_group_modules.modules);
        self.add_module_item(root_item_path, root_items);

        Ok(())
    }
}

fn shader_root_module<F>(
    module: &naga::Module,
    wgsl_source: &str,
    wgsl_include_path: Option<&str>,
    root_path: ModulePath,
    bind_group_data: &BTreeMap<u32, bindgroup::GroupData<'_>>,
    demangle: F,
    bind_groups_module: TokenStream,
) -> (TypePath, TokenStream)
where
    F: Fn(&str) -> TypePath + Clone,
{
    let compute_module = compute_module(module, demangle.clone());
    let fragment_states = fragment_states(module, demangle.clone());

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
        .iter()
        .map(|(group_no, group)| {
            let path = match &group.name {
                GroupName::Module(module) => root_path.relative_module_path(module),
                _ => quote! {bind_groups},
            };

            let group_type = group.camel_case_ident("BindGroup", *group_no);
            quote!(#path::#group_type::get_bind_group_layout(device))
        })
        .collect();

    let immediate_size = immediate_data_size(module).unwrap_or(quote!(0));

    let create_pipeline_layout = quote! {
        pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    #(&#bind_group_layouts),*
                ],
                immediate_size: #immediate_size,
            })
        }
    };

    let override_constants = pipeline_overridable_constants(module, demangle);

    let set_bind_groups = if !bind_group_data.is_empty() {
        Some(set_bind_groups_func(bind_group_data, &root_path))
    } else {
        None
    };

    // Place items in the root module for this shader.
    let root_item_path = TypePath {
        parent: root_path.clone(),
        name: String::new(),
    };
    let root_items = quote! {
        #override_constants
        #bind_groups_module
        #set_bind_groups
        #compute_module
        #fragment_states
        #create_shader_module
        #create_pipeline_layout
    };
    (root_item_path, root_items)
}

fn shared_root_module(
    bind_group_data: &BTreeMap<u32, bindgroup::GroupData<'_>>,
    vertex_states: &[(TypePath, TokenStream)],
) -> (TypePath, TokenStream) {
    // Add shared code to the top level module to use with all shader modules.
    let mut shared_items = quote!();
    if !vertex_states.is_empty() {
        let vertex_states_shared = vertex_states_shared();
        shared_items.extend(vertex_states_shared);
    }
    if !bind_group_data.is_empty() {
        let bind_groups_root = set_bind_groups_trait();
        shared_items.extend(bind_groups_root);
    }

    // The type path name here just needs to be unique.
    let shared_path = TypePath {
        parent: ModulePath::default(),
        name: "__SHARED".to_string(),
    };
    (shared_path, shared_items)
}

fn immediate_data_size(module: &naga::Module) -> Option<TokenStream> {
    // Assume only one variable is used with var<immediate> in WGSL.
    let (_, global) = module
        .global_variables
        .iter()
        .find(|(_, g)| g.space == naga::AddressSpace::Immediate)?;

    let size = module.types[global.ty].inner.size(module.to_ctx());

    // wgpu only uses a single immediate data range for all shader stages.
    // This allows easily setting immediate data in a single call with offset 0.
    let size = Literal::usize_unsuffixed(size as usize);
    Some(quote!(#size))
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
        if stages.contains(wgpu::ShaderStages::TASK) {
            components.push(quote!(wgpu::ShaderStages::TASK));
        }
        if stages.contains(wgpu::ShaderStages::MESH) {
            components.push(quote!(wgpu::ShaderStages::MESH));
        }

        if let Some((first, remaining)) = components.split_first() {
            quote!(#first #(.union(#remaining))*)
        } else {
            quote!(wgpu::ShaderStages::NONE)
        }
    }
}

fn demangle_with_root<F: Fn(&str) -> TypePath + Clone>(
    f: F,
    root_path: ModulePath,
) -> impl Fn(&str) -> TypePath + Clone {
    move |path| {
        let mut path = f(path);
        if path.parent.components.is_empty() {
            path.parent = root_path.clone();
        }
        path
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use indoc::indoc;

    // Tokenstreams can't be compared directly using PartialEq.
    // Use pretty_print to normalize the formatting and compare strings.
    // Use a colored diff output to make differences easier to see.
    #[cfg(test)]
    #[macro_export]
    macro_rules! assert_tokens_eq {
        ($a:expr, $b:expr) => {
            pretty_assertions::assert_eq!(
                $crate::pretty_print_rustfmt($a),
                $crate::pretty_print_rustfmt($b)
            )
        };
    }

    #[cfg(test)]
    #[macro_export]
    macro_rules! assert_tokens_snapshot {
        ($output:expr) => {
            let mut settings = insta::Settings::new();
            settings.set_prepend_module_to_snapshot(false);
            settings.set_omit_expression(true);
            settings.bind(|| {
                insta::assert_snapshot!($crate::pretty_print_rustfmt($output));
            });
        };
    }

    #[cfg(test)]
    #[macro_export]
    macro_rules! assert_rust_snapshot {
        ($output:expr) => {
            let mut settings = insta::Settings::new();
            settings.set_prepend_module_to_snapshot(false);
            settings.set_omit_expression(true);
            settings.bind(|| {
                insta::assert_binary_snapshot!(".rs", $output.into());
            });
        };
    }

    #[test]
    fn create_shader_module_immediate_data() {
        let source = indoc! {r#"
            var<immediate> consts: vec4<f32>;

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return consts;
            }
        "#};

        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
        assert_rust_snapshot!(actual);
    }

    #[test]
    fn create_shader_multiple_entries() {
        // Test all shader stages.
        let source = include_str!("data/entries_all_stages.wgsl");
        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
        assert_rust_snapshot!(actual);
    }

    #[test]
    fn create_shader_module_multiple_outputs() {
        let source = indoc! {r#"
            struct Output {
                @location(0) col0: vec4<f32>,
                @builtin(frag_depth) depth: f32,
                @location(1) col1: vec4<f32>,
            };

            @fragment
            fn fs_multiple() -> Output {}
        "#};
        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
        assert_rust_snapshot!(actual);
    }

    #[test]
    fn create_shader_module_compute_overrides() {
        let source = indoc! {r#"
            struct Uniforms {
                color_rgb: vec3<f32>,
            }

            @group(0) @binding(0)
            var<storage, read_write> uniforms: Uniforms;

            override dt: f32;

            @compute
            @workgroup_size(1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x == 0u {
                    uniforms.color_rgb = vec3(1.0+dt);
                }
            }
        "#};
        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
        assert_rust_snapshot!(actual);
    }

    #[test]
    fn create_shader_module_overrides() {
        let source = indoc! {r#"
            override b1: bool = true;
            override b2: bool = false;
            override b3: bool;

            override f1: f32 = 0.5;
            override f2: f32;

            override f3: f64 = 0.6;
            override f4: f64;

            override i1: i32 = 0;
            override i2: i32;
            override i3: i32 = i1 * i2;

            @id(0) override a: f32 = 1.0;
            @id(35) override b: f32 = 2.0;

            @fragment
            fn main() {}
        "#};
        let actual = create_shader_module(source, "shader.wgsl", WriteOptions::default()).unwrap();
        assert_rust_snapshot!(actual);
    }

    #[test]
    fn create_shader_modules_source() {
        let source = "@fragment fn main() {}";
        let actual =
            create_shader_modules(source, WriteOptions::default(), demangle_identity).unwrap();
        assert_rust_snapshot!(actual);
    }

    #[test]
    fn create_shader_modules_source_rustfmt() {
        let source = "@fragment fn main() {}";
        let actual = create_shader_modules(
            source,
            WriteOptions {
                rustfmt: true,
                ..Default::default()
            },
            demangle_identity,
        )
        .unwrap();
        assert_rust_snapshot!(actual);
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
        root.add_module_items(items);
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

        assert_tokens_snapshot!(items_to_tokens(actual));
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

        assert_tokens_snapshot!(items_to_tokens(actual));
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

        assert_tokens_snapshot!(items_to_tokens(actual));
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

        assert_tokens_snapshot!(items_to_tokens(actual));
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

        assert_tokens_snapshot!(items_to_tokens(actual));
    }

    #[test]
    fn create_shader_module_parse_error() {
        let source = indoc! {r#"
            var<immediate> consts: vec4<f32>;

            @fragment
            fn fs_main() }
        "#};

        let result = create_shader_module(source, "shader.wgsl", WriteOptions::default());

        assert!(matches!(result, Err(CreateModuleError::ParseError { .. })));
    }

    #[test]
    fn create_shader_module_semantic_error() {
        let source = indoc! {r#"
            var<immediate> consts: vec4<f32>;

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

        assert!(matches!(
            result,
            Err(CreateModuleError::ValidationError { .. })
        ));
    }

    fn demangle_underscore(name: &str) -> TypePath {
        // Preprocessors that support modules mangle absolute paths.
        // Use a very basic mangling scheme that assumes no '_' in the identifier name.
        // This allows testing the module logic without needing extra dependencies.
        // a_b_C -> a::b::C
        let components: Vec<_> = name.split("_").collect();
        let (name, parents) = components.split_last().unwrap();
        TypePath {
            parent: ModulePath {
                components: parents.iter().map(|p| p.to_string()).collect(),
            },
            name: name.to_string(),
        }
    }

    #[test]
    fn single_root_module() {
        let output = create_shader_modules(
            include_str!("data/modules.wgsl"),
            WriteOptions {
                rustfmt: true,
                ..Default::default()
            },
            demangle_underscore,
        )
        .unwrap();

        assert_rust_snapshot!(output);
    }

    #[test]
    fn add_single_root_module() {
        let mut root = Module::default();
        let options = WriteOptions {
            rustfmt: true,
            ..Default::default()
        };
        root.add_shader_module(
            include_str!("data/modules.wgsl"),
            None,
            options,
            ModulePath::default(),
            demangle_underscore,
        )
        .unwrap();

        let output = root.to_generated_bindings(options);
        assert_rust_snapshot!(output);
    }

    #[test]
    fn add_duplicate_module_different_paths() {
        // Test shared types and handling of duplicate names.
        let mut root = Module::default();
        let options = WriteOptions {
            rustfmt: true,
            ..Default::default()
        };
        root.add_shader_module(
            include_str!("data/modules.wgsl"),
            None,
            options,
            ModulePath {
                components: vec!["shader1".to_string()],
            },
            demangle_underscore,
        )
        .unwrap();
        root.add_shader_module(
            include_str!("data/modules.wgsl"),
            None,
            options,
            ModulePath {
                components: vec!["shaders".to_string(), "shader2".to_string()],
            },
            demangle_underscore,
        )
        .unwrap();

        let output = root.to_generated_bindings(options);
        assert_rust_snapshot!(output);
    }

    #[test]
    fn vertex_entries() {
        // Check vertex entry points and builtin attribute handling.
        let actual = create_shader_module(
            include_str!("data/vertex_entries.wgsl"),
            "shader.wgsl",
            WriteOptions {
                rustfmt: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_rust_snapshot!(actual);
    }

    #[test]
    fn shader_stage_collection() {
        // Check the visibility: wgpu::ShaderStages::COMPUTE
        let actual = create_shader_module(
            include_str!("data/shader_stage_collection.wgsl"),
            "shader.wgsl",
            WriteOptions {
                rustfmt: true,
                derive_encase_host_shareable: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_rust_snapshot!(actual);
    }

    #[test]
    fn mesh_shader() {
        let output = create_shader_modules(
            include_str!("data/mesh_shader.wgsl"),
            WriteOptions {
                rustfmt: true,
                ..Default::default()
            },
            demangle_identity,
        )
        .unwrap();

        assert_rust_snapshot!(output);
    }

    macro_rules! bind_group_named_tests {
        (
        	Settings { named: $named:expr, shared: $shared:expr },
        	$(($test_name:ident, $file_name:expr),)*
        ) => {
        	$(
	         	#[test]
		         fn $test_name() {
		             let output = create_shader_modules(
                		include_str!(concat!("data/bindgroup/named_", $file_name, ".wgsl")),
		                 WriteOptions {
		                     rustfmt: true,
		                     named_bind_groups: $named,
                             shared_bind_groups: $shared,
		                     ..Default::default()
		                 },
		                 demangle_underscore,
		             )
		             .unwrap();

		             assert_rust_snapshot!(output);
				}
         	)*
        };
    }

    bind_group_named_tests!(
        Settings {
            named: true,
            shared: false
        },
        (bind_group_named_simple, "simple"),
        (bind_group_named_multiple_bindings, "multiple_bindings"),
        (bind_group_named_different_modules, "different_modules"),
    );

    bind_group_named_tests!(
        Settings {
            named: false,
            shared: true
        },
        (bind_group_shared_simple, "simple"),
        (bind_group_shared_multiple_bindings, "multiple_bindings"),
        (bind_group_shared_different_modules, "different_modules"),
    );
}
