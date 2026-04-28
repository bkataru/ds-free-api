# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.3] - 2026-04-28

### Added
- `tools_present` compatibility: accept and normalize client tool-presence signaling so adapters stay aligned with OpenAI-shaped clients that set the flag explicitly.
- Reasoning-merge in `openai_adapter` response conversion: merge interim reasoning deltas with assistant content so streaming and aggregated outputs stay coherent across tool and text phases.
- `pending_reasoning_flush` handling: flush deferred reasoning fragments at stream boundaries so reasoning does not leak across chunks or get dropped between states.
- Converter trace instrumentation: structured trace points in the OpenAI SSE converter path to debug frame ordering, reasoning vs. content splits, and tool-call interleaving.


### Fixed
- OMP XML tool-call parsing: recognize `<tool_calls><invoke ...>` output and convert it into OpenAI-compatible structured `tool_calls` instead of assistant text.
- Tool-call repair resilience: parse repaired JSON directly before XML fallback so docs or tool arguments containing literal `<tool_calls>` markers do not break repair.
- Reasoning-only tool calls: preserve tool calls emitted from DeepSeek thinking output when the upstream state machine finishes without a separate response body.

## [0.2.2] - 2026-04-22

### Added
- Anthropic Messages API compatibility layer:
  - `/anthropic/v1/messages` streaming and non-streaming endpoints
  - `/anthropic/v1/models` list/get endpoints (Anthropic response shape)
  - Request mapping: Anthropic JSON → OpenAI ChatCompletion
  - Response mapping: OpenAI SSE/JSON → Anthropic Message SSE/JSON
- OpenAI adapter backward compatibility:
  - Deprecated `functions` / `function_call` automatically mapped to `tools` / `tool_choice`
  - `response_format` downgrade: JSON/Schema constraints injected into the ChatML prompt (`text` type is a no-op)
- CI release workflow improvements:
  - Tag-triggered releases (`push.tags v*`)
  - Automatic extraction of release notes from the changelog
  - Pre-release check that `Cargo.toml` version matches the tag

### Changed
- Rust toolchain bumped to 1.95.0; CI workflows updated accordingly
- `justfile`: add `set positional-arguments` so arguments containing spaces pass through safely
- Python E2E tests reorganized into `openai_endpoint/` and `anthropic_endpoint/`
- Startup logs print OpenAI and Anthropic base URLs
- README / README.en.md: SVG icons, GitHub badges, docs kept in sync
- LICENSE: copyright notice `Copyright 2026 NIyueeE`
- CLAUDE.md / AGENTS.md synchronized

### Fixed
- Anthropic streaming tool-call protocol: use `input_json_delta` events to stream tool parameters incrementally
- Tool use ID mapping consistency: `call_{suffix}` → `toolu_{suffix}`
- Anthropic tool-definition compatibility: handle tool specs missing a `type` field (Claude Code client)

## [0.2.1] - 2026-04-15

### Added
- Reasoning on by default: `reasoning_effort` defaults to `high`; web search defaults off.
- Dynamic WASM discovery: `pow.rs` uses signature-based export detection instead of hard-coding `__wbindgen_export_0`, reducing startup failures when DeepSeek ships a new WASM build.
- Python E2E test suite covering auth, models, chat completions, tool calling, and related scenarios.
- `tiktoken-rs` dependency for server-side prompt token estimation.
- CI: `cargo audit` and `cargo machete` checks.

### Changed
- Account initialization: logs fall back from phone number to email when the phone field is empty.
- Core dependencies (`axum`, `cranelift`, and others) bumped to latest patch releases.
- Client Version kept aligned with the web UI at `1.8.0`.

### Removed
- Unused `tower` dependency.

## [0.2.0] - 2026-04-13

### Added
- Full Rust rewrite replacing the Python implementation for native performance and cross-platform binaries.
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`).
- Account-pool rotation, PoW solving, and SSE streaming responses.
- Deep reasoning and web search support.
- Tool calling via XML parsing.
- GitHub CI and multi-platform releases (eight target triples).
- Compatibility with the then-current DeepSeek web backend.
