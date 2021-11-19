use npm_rs::NpmEnv;

fn main() {
    let status = NpmEnv::default()
        .set_path("frontend")
        .init_env()
        .install(None)
        .run("build")
        .exec()
        .unwrap();
    assert!(status.success());
}
