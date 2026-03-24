#[test]
fn test_compile() {
    // Don't include the generated modules here directly.
    // This allows projects to compile even with invalid generated code.
    // TODO: this test failure skips other tests unless using --no-fail-fast
    // TODO: generate a test.pass for each file instead?
    // TODO: write file to OUT_DIR for each snapshot with a dummy main function.
    let t = trybuild::TestCases::new();
    t.pass("src/compile_tests.rs");
}
