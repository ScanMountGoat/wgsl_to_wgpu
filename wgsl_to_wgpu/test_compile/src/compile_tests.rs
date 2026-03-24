#![allow(clippy::unnecessary_cast, nonstandard_style)]
include!(concat!(env!("OUT_DIR"), "/import_snapshots.rs"));

// trybuild requires a main function.
fn main() {}
