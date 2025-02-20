use std::path::PathBuf;

#[allow(dead_code)]
fn run_mode(mode: &'static str) {
    let mut config = compiletest_rs::Config::default();

    config.mode = mode.parse().expect("Invalid mode");
    config.src_base = PathBuf::from(format!("tests/{}", mode));
    config.link_deps();
    config.clean_rmeta();

    compiletest_rs::run_tests(&config);
}

#[test]
fn compile_test() {
    #[cfg(target_os = "linux")]
    run_mode("compile-fail");
}
