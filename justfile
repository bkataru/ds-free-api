# Justfile for ai-free-api

set positional-arguments

# Run all checks: type check, lint, format, audit, unused deps
# 前置: cargo install cargo-audit && cargo install cargo-machete
check:
  cargo check
  cargo clippy -- -D warnings
  cargo fmt --check
  cargo audit
  cargo machete

# Run ds_core_cli example
ds-core-cli *ARGS:
  cargo run --example ds_core_cli -- "$@"

# Run openai_adapter/request submodule tests
test-adapter-request *ARGS:
  cargo test openai_adapter::request -- "$@"

# Run openai_adapter/response submodule tests
test-adapter-response *ARGS:
  cargo test openai_adapter::response -- "$@"

# Run openai_adapter_cli example
openai-adapter-cli *ARGS:
  cargo run --example openai_adapter_cli -- "$@"

# Run HTTP server
serve *ARGS:
  cargo run -- "$@"

# Run Python e2e tests (requires server running; will skip with hint if not)
# -n 2: 并发测试（DeepSeek 免费 API 不支持更高并发，4 workers 会触发大量空响应）
e2e *ARGS:
  cd py-e2e-tests && uv run python -m pytest -n 2 "$@"

# Start server with e2e test config
e2e-serve:
  cargo run -- -c py-e2e-tests/config.toml
