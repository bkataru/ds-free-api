# Logging

## Principles

1. **Libraries stay silent**: modules under `ds_core/` log only via the `log` crate — never write directly to stdout/stderr.
2. **Callers configure output**: levels, formats, and sinks live in binaries (`main.rs`, examples).
3. **Structured targets**: module filtering uses target paths (`crate::module`).

## Levels

| Level | When | Examples |
|------|------|-----------|
| `ERROR` | Needs human intervention | All accounts failed to initialize, PoW module panic, invalid config |
| `WARN` | Degraded path, still workable | Single account failed (others remain), session cleanup failed |
| `INFO` | Lifecycle milestones | Account ready, server start/stop, session created |
| `DEBUG` | Diagnostics | HTTP request/response summaries, PoW timing, SSE event names |
| `TRACE` | Maximum detail | Raw SSE bytes, full HTTP bodies |

## Targets

Format: `crate::module` or `crate::module::submodule`

| Module | Target | Covers |
|--------|--------|--------|
| `ds_core::accounts` | `ds_core::accounts` | Pool lifecycle, health checks |
| `ds_core::client` | `ds_core::client` | HTTP plumbing, API calls |
| `ds_core::completions` | `ds_core::completions` | Turn orchestration, SSE handling |
| `ds_core::pow` | `ds_core::pow` | PoW runs, WASM load |
| `adapter` | `adapter` | OpenAI compatibility layer |

## Usage

### Libraries (`ds_core/`)

```rust
use log::{info, debug, warn, error};

// INFO — lifecycle checkpoints
info!(target: "ds_core::accounts", "account {} ready", mobile);

// WARN — one failure among many (degrade gracefully)
warn!(target: "ds_core::accounts", "account {} init failed: {}", mobile, e);

// DEBUG — transport / protocol detail
debug!(target: "ds_core::client", "PoW challenge: alg={} difficulty={}", alg, diff);

// ERROR — fatal condition
error!(target: "ds_core::accounts", "all accounts failed to initialize");
```

### Binaries (`examples/`, `main.rs`)

```rust
fn main() {
    // Default `info`; override with `RUST_LOG`
    env_logger::Builder::from_env(
        env_logger::Env::new().default_filter_or("info")
    ).init();
}
```

## Runtime tuning

```bash
# Default (info)
cargo run --example ds_core_cli

# Verbose — all debug lines
RUST_LOG=debug cargo run --example ds_core_cli

# Targeted — only accounts at debug
RUST_LOG=ds_core::accounts=debug cargo run --example ds_core_cli

# Mixed — accounts debug, client warn, rest info
RUST_LOG=ds_core::accounts=debug,ds_core::client=warn,info cargo run --example ds_core_cli

# Errors only
RUST_LOG=error cargo run --example ds_core_cli

# Capture stderr to a file
RUST_LOG=debug cargo run --example ds_core_cli 2> ds_core.log
```

## Do not

- Use `println!` / `eprintln!` inside library crates
- Call `info!`-style macros without an explicit `target`
- Emit secrets (tokens, passwords)
- Enable high-frequency TRACE (per-byte SSE dumps) globally by default

## Dependencies

**Cargo.toml**

```toml
[dependencies]
log = "0.4"

[dev-dependencies]
env_logger = { version = "0.11", default-features = false, features = ["auto-color"] }
```

Note: `auto-color` colors TTY streams and disables styling when stdout is not a TTY.
