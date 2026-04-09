# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Rust API proxy exposing free DeepSeek (and eventually Qwen) model endpoints. Translates standard OpenAI-compatible requests to provider-specific protocols with account pool rotation, PoW challenge handling, and streaming response support.

## Principles

### 1. Single Responsibility
- `config.rs`: Configuration loading only, no client creation or business logic
- `client.rs`: Raw HTTP calls only, no token caching, retry, or SSE parsing
- `accounts.rs`: Account pool management only, no network requests
- `pow.rs`: WASM computation only, no account management or request sending

### 2. Minimal Viable
- No premature abstractions: Extract traits/structs when needed, not before
- No redundant code: Remove unused imports, avoid over-documenting, no pre-written tests
- Delay dependency introduction: pingora temporarily removed, will add when proxy layer needed

### 3. Control Complexity
- Explicit over implicit: Dependencies injected via parameters, no global state
- Composition over inheritance: Small modules composed via functions, no deep inheritance
- Clear boundaries: Modules interact via explicit interfaces, no internal logic leakage

## Architecture

```
src/
├── main.rs          # Entry point (stub)
├── lib.rs           # Library exports
├── config.rs        # Config loader: -c flag, ../config.toml default
├── adapter.rs       # OpenAI→provider adapter layer (empty)
├── adapter/         # Adapter implementations (empty)
├── ds_core.rs       # DeepSeek module facade (v0_chat entry)
├── ds_core/
│   ├── accounts.rs  # Account pool: init validation, round-robin selection
│   ├── pow.rs       # PoW solver: WASM loading, DeepSeekHashV1 computation
│   ├── completions.rs # Chat orchestration: SSE streaming, account guard
│   └── client.rs    # Raw HTTP client: API endpoints, zero business logic
└── qw_core.rs       # Qwen module stub (empty)
```

**Account Pool Model**: 1 account = 1 session = 1 concurrency. Scale via more accounts.

**Request Flow**: `v0_chat()` → `get_account()` → `compute_pow()` → `edit_message()` → SSE stream

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Config loading | `src/config.rs` | Single unified entry, `-c` flag support |
| DeepSeek chat flow | `src/ds_core/` | accounts → pow → completions → client |
| Provider abstraction | `src/adapter.rs` | Planned OpenAI-compatible adapter |
| Logging spec | `docs/logging-spec.md` | Target naming, levels, examples |
| Code style | `docs/code-style.md` | Comments, naming, imports |
| API reference | `docs/deepseek-api-reference.md` | DeepSeek endpoint details |

## Conventions

- **Config**: Uncommented values in `config.toml` = required; commented = optional with default
- **Module files**: `foo.rs` declares sub-modules, `foo/` contains implementation
- **Comments**: Chinese in source files (team preference)
- **Errors**: Chinese error messages for user-facing output

## Anti-Patterns

- Do NOT create separate config entry points — `src/config.rs` is the single source
- Do NOT implement provider logic outside its `*_core/` module
- Do NOT commit `config.toml` (only `config.toml.example`)
- Do NOT use `println!`/`eprintln!` in library code — use `log` crate with target

## Commands

```bash
# One-pass check (check + clippy + fmt)
just check

# Run ds_core_cli example
just ds-core-cli
RUST_LOG=debug just ds-core-cli
just ds-core-cli -- source test.txt  # script mode

# Individual checks
cargo check
cargo clippy -- -D warnings
cargo fmt --check

# Build and test
cargo build
cargo test
```

## Implementation Status

- `ds_core::client` ✅ — HTTP layer complete
- `ds_core::pow` ✅ — WASM PoW solver
- `ds_core::accounts` ✅ — Account pool with lifecycle management
- `ds_core::completions` ✅ — SSE streaming with GuardedStream
- `main.rs` ⚠️ — Stub only, server bootstrap pending
- `adapter` ⚠️ — Not started
- `qw_core` ⚠️ — Not started
