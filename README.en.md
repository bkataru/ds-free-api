<p align="center">
  <img src="https://raw.githubusercontent.com/NIyueeE/ds-free-api/main/assets/logo.svg" width="81" height="66">
</p>

<h1 align="center">DS-Free-API</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/NIyueeE/ds-free-api.svg"></a>
  <img src="https://img.shields.io/github/v/release/NIyueeE/ds-free-api.svg">
  <img src="https://img.shields.io/badge/rust-1.95.0+-93450a.svg">
  <img src="https://github.com/NIyueeE/ds-free-api/actions/workflows/ci.yml/badge.svg">
</p>
<p align="center">
  <img src="https://img.shields.io/github/stars/NIyueeE/ds-free-api.svg">
  <img src="https://img.shields.io/github/forks/NIyueeE/ds-free-api.svg">
  <img src="https://img.shields.io/github/last-commit/NIyueeE/ds-free-api.svg">
  <img src="https://img.shields.io/github/languages/code-size/NIyueeE/ds-free-api.svg">
</p>

[中文](README.md)

Reverse proxy and adapter for free DeepSeek web chat endpoints to standard OpenAI-compatible and Anthropic-compatible API protocols (currently supports chat completions and messages, including streaming and tool calls).

Cross-platform native Rust binary + single TOML config file.

## Quick Start

Download the release for your platform from [releases](https://github.com/NIyueeE/ds-free-api/releases) and extract.

```
  .
  ├── ds-free-api          # executable
  ├── LICENSE
  ├── README.md
  ├── README.en.md
  └── config.example.toml  # config template
```

### Configuration

Copy `config.example.toml` to `config.toml` in the same directory as the executable, or use `./ds-free-api -c <config_path>` to specify a config path.

### Run

```bash
# Run directly (requires config.toml in the same directory)
./ds-free-api

# Specify config path
./ds-free-api -c /path/to/config.toml

# Debug mode
RUST_LOG=debug ./ds-free-api
```

Required fields only. One account = one concurrency slot (seems max 2 concurrent).

```toml
[server]
host = "127.0.0.1"
port = 5317

# API access tokens, leave empty to disable auth
# [[server.api_tokens]]
# token = "sk-your-token"
# description = "dev test"

# Fill email or mobile (pick one or both). Mobile seems to only support +86 area.
[[accounts]]
email = "user1@example.com"
mobile = ""
area_code = ""
password = "pass1"
```

Here are some free test accounts — please don't send sensitive info through them (the program deletes sessions on cleanup, but leftovers may persist).

```text
rivigol378@tatefarm.com
test12345

counterfeit1341@wplacetools.com
test12345

idyllic4202@wplacetools.com
test12345

slowly1285@wplacetools.com
test12345
```

If you want multiple accounts for concurrency, look into temporary email services (some may not work) and use a VPN to register on the international version.

Recommended temp-mail site: [tempmail.la](https://tempmail.la/) (some domains may not work, try a few times)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/v1/chat/completions` | Chat completions (streaming and tool calls supported) |
| GET | `/v1/models` | List models |
| GET | `/v1/models/{id}` | Get model |
| POST | `/anthropic/v1/messages` | Anthropic Messages API (streaming and tool calls supported) |
| GET | `/anthropic/v1/models` | List models (Anthropic format) |
| GET | `/anthropic/v1/models/{id}` | Get model (Anthropic format) |

## Model Mapping

`model_types` in `config.toml` (defaults to `["default", "expert"]`) maps automatically:

| OpenAI Model ID | DeepSeek Type |
|-----------------|---------------|
| `deepseek-default` | Default mode |
| `deepseek-expert` | Expert mode |

The Anthropic compatibility layer uses the same model IDs, accessed via `/anthropic/v1/messages`.

### Capability Toggles

- **Reasoning**: Enabled by default. To disable, add `"reasoning_effort": "none"` to the request body.
- **Web search**: Disabled by default. To enable, add `"web_search_options": {"search_context_size": "high"}`.
- **Tool calls**: Pass standard OpenAI `tools` and `tool_choice` fields. When the model decides to call a tool, the returned `finish_reason` will be `tool_calls`.

## Development

Requires Rust 1.95.0+ (see `rust-toolchain.toml`).

```bash
# One-pass check (check + clippy + fmt + audit + unused deps)
just check

# Run tests
cargo test

# Run HTTP server
just serve

# CLI examples
just ds-core-cli
just openai-adapter-cli

# Python e2e tests (requires server running on port 5317)
just e2e

# Start server with e2e test config
just e2e-serve
```

Architecture overview:

```mermaid
flowchart TB
    %% ========== Style definitions ==========
    classDef client fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#1e40af,rx:12,ry:12
    classDef gateway fill:#fef9c3,stroke:#ca8a04,stroke-width:3px,color:#854d0e,rx:10,ry:10
    classDef adapter fill:#fae8ff,stroke:#a21caf,stroke-width:2px,color:#701a75,rx:8,ry:8
    classDef core fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#14532d,rx:8,ry:8
    classDef external fill:#fee2e2,stroke:#dc2626,stroke-width:3px,color:#991b1b,rx:4,ry:4

    %% ========== Node definitions ==========
    Client(["Client<br/>OpenAI / Anthropic protocol"]):::client

    subgraph GW ["Gateway (axum HTTP)"]
        Server(["Routing / Auth"]):::gateway
    end

    subgraph PL ["Protocol Layer"]
        direction LR

        subgraph AC ["Anthropic compat (anthropic_compat)"]
            A2OReq["Anthropic → OpenAI<br/>request mapping"]:::adapter
            O2AResp["OpenAI → Anthropic<br/>response mapping"]:::adapter
        end

        subgraph OA ["OpenAI adapter (openai_adapter)"]
            ReqParse["Request parsing"]:::adapter
            RespTrans["Response conversion"]:::adapter
        end
    end

    subgraph CL ["Core logic (ds_core)"]
        direction TB
        Pool["Account pool rotation"]:::core
        Pow["PoW solving"]:::core
        Chat["Chat orchestration"]:::core
    end

    DeepSeek[("DeepSeek API")]:::external

    %% ========== Request flow ==========
    Client -->|"OpenAI / Anthropic protocol"| Server
    Server -->|"OpenAI traffic"| ReqParse
    Server -->|"Anthropic traffic"| A2OReq
    A2OReq --> ReqParse
    ReqParse --> Pool
    Pool --> Pow
    Pow --> Chat
    Chat -->|"DeepSeek internal protocol"| DeepSeek

    %% ========== Response flow ==========
    Chat -.->|"SSE stream"| RespTrans
    RespTrans -.->|"OpenAI response"| Server
    RespTrans -.->|"pending conversion"| O2AResp
    O2AResp -.->|"Anthropic response"| Server

    %% ========== Subgraph styling ==========
    style GW fill:#fefce8,stroke:#eab308,stroke-width:2px
    style PL fill:#fafafa,stroke:#a3a3a3,stroke-width:2px
    style AC fill:#fdf4ff,stroke:#c026d3,stroke-width:1px
    style OA fill:#f5f3ff,stroke:#8b5cf6,stroke-width:1px
    style CL fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
```

Data pipelines:

- **OpenAI request**: `JSON body` → `normalize` validation/defaults → `tools` extraction → `prompt` ChatML build → `resolver` model mapping → `ChatRequest`
- **OpenAI response**: `DeepSeek SSE bytes` → `sse_parser` → `state` patch state machine → `converter` format conversion → `tool_parser` XML parsing → `StopStream` truncation → `OpenAI SSE bytes`
- **Anthropic request**: `Anthropic JSON` → `to_openai_request()` → enters OpenAI request pipeline
- **Anthropic response**: OpenAI output → `from_chat_completion_stream()` / `from_chat_completion_bytes()` → `Anthropic SSE/JSON`

## License

[Apache License 2.0](LICENSE)

DeepSeek's official API is very affordable. Please support the official service.

This project was created to experiment with the latest models in DeepSeek's web A/B testing.

**Commercial use is strictly prohibited** to avoid putting pressure on official servers. Use at your own risk.
