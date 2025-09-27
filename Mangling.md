# Name Mangling
Name mangling is necessary in some cases to uniquely identify items and ensure valid WGSL names. The `demangle` function supplied to wgsl_to_wgpu converts mangled absolute module paths to module path components.

The following sections describe `demangle` functions for various popular WGSL preprocessing libraries. Please submit an issue or pull request if a library is not listed here or the provided implementation is inaccurate.

## Wesl
Wesl supports multiple mangling schemes. Make sure that the demangle function uses the mangler configured in the options.

Wesl does not mangle identifiers in the root module by default. wgsl_to_wgpu demangles all identifiers, which can cause issues demangling certain names. A workaround is to detect if a name is mangled by looking for the appropriate prefix.

```rust
// wesl 0.2.0
fn demangle_wesl(name: &str) -> wgsl_to_wgpu::TypePath {
    // Assume all paths are absolute paths.
    if name.starts_with("package_") {
        // Use the root module if unmangle fails.
        let mangler = wesl::EscapeMangler;
        let (path, name) = mangler
            .unmangle(name)
            .unwrap_or((wesl::ModulePath::default(), name.to_string()));

        // Assume all wesl paths are absolute paths.
        wgsl_to_wgpu::TypePath {
            parent: wgsl_to_wgpu::ModulePath {
                components: path.components,
            },
            name,
        }
    } else {
        // Use the root module if the name is not mangled.
        wgsl_to_wgpu::TypePath {
            parent: wgsl_to_wgpu::ModulePath::new_root(),
            name: name.to_string(),
        }
    }
}
```

## naga_oil
The function `naga_oil::compose::undecorate` is private and will need to be implemented manually. This requires additional dependencies like `data-encoding`, `regex`, and `regex-syntax`. Most projects should consider using wesl due to its easier integration with wgsl_to_wgpu.

```rust
fn demangle_naga_oil(name: &str) -> wgsl_to_wgpu::TypePath {
    // fn undecorate(&self, string: &str) -> String
    let name = undecorate(name);
    let parts: Vec<_> = name.split("::").collect();
    let (name, parents) = parts.split_last().unwrap();

    wgsl_to_wgpu::TypePath {
        parent: wgsl_to_wgpu::ModulePath {
            components: parents.into_iter().map(|s| s.to_string()).collect(),
        },
        name: name.to_string(),
    }
}
```
