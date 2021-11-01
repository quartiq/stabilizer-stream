fn main() {
    println!("cargo-rerun-if-changed=frontend");

    let mut working_directory = std::env::current_dir().unwrap();
    working_directory.push("frontend");
    assert!(
        npm_rs::NpmEnv::default()
            .set_path(working_directory)
            .init_env()
            .run("build")
            .exec()
            .unwrap()
            .success(),
        "Failed to build front-end resources"
    );
}
