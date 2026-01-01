use std::{ffi::OsStr, path::Path};

const PATH: &str = "../src/snapshots";

/// Create a file, which imports all snapshots in `PATH`.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", PATH);

    let mut import_snapshots = String::new();
    for entry in std::fs::read_dir(PATH).unwrap() {
        let path = entry.unwrap().path().canonicalize().unwrap();

        if path.extension() != Some(OsStr::new("rs")) {
            continue;
        }

        let name = path.file_prefix().unwrap().to_str().unwrap();
        let module = format!("#[path = \"{}\"] pub mod {};\n", path.display(), name);
        import_snapshots.push_str(&module);
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("import_snapshots.rs");
    std::fs::write(&out_path, import_snapshots).unwrap();
}
