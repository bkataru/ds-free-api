# Justfile for ai-free-api

# Run all checks: type check, lint, format
check:
  cargo check
  cargo clippy -- -D warnings
  cargo fmt --check

# Run ds_core_cli example
ds-core-cli *ARGS:
  cargo run --example ds_core_cli -- {{ARGS}}

# Run openai_adapter/request submodule tests
test-adapter-request *ARGS:
  cargo test openai_adapter::request -- {{ARGS}}

# Run openai_adapter/response submodule tests
test-adapter-response *ARGS:
  cargo test openai_adapter::response -- {{ARGS}}
