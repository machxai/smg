fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild when the vendored contract changes.
    println!("cargo:rerun-if-changed=proto/inferencecache/v1alpha1/inferencecache.proto");

    // Build both client and server: the client is the production surface, the
    // server stubs are used by the in-process mock in the unit tests.
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &["proto/inferencecache/v1alpha1/inferencecache.proto"],
            &["proto"],
        )?;

    Ok(())
}
