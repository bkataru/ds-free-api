# Justfile for ai-free-api

# Run all checks: type check, lint, format
check:
  cargo check
  cargo clippy -- -D warnings
  cargo fmt --check

# Run ds_core_cli example
ds-core-cli *ARGS:
  cargo run --example ds_core_cli -- {{ARGS}}
