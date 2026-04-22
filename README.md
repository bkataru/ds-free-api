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

[English](README.en.md)

将免费的 DeepSeek 网页端对话反代并适配转换为标准的 OpenAI 与 Anthropic 兼容 API 协议（目前支持 chat completions 和 messages，包括流式返回与工具调用）。

支持 Rust 原生多端高性能，单可执行文件 + 单 TOML 配置文件。

## 快速开始

去 [releases](https://github.com/NIyueeE/ds-free-api/releases) 下载对应平台后解压即可。

```
  .
  ├── ds-free-api          # 可执行文件
  ├── LICENSE
  ├── README.md
  ├── README.en.md
  └── config.example.toml  # 配置示例
```

### 配置

复制 `config.example.toml` 为 `config.toml`，和可执行文件保持在同一个路径下，或者使用 `./ds-free-api -c <config_path>` 指定配置路径。

### 运行

```bash
# 直接运行 (同目录下需要 config.toml)
./ds-free-api

# 指定配置路径
./ds-free-api -c /path/to/config.toml

# 调试模式
RUST_LOG=debug ./ds-free-api
```

这里只展示必填项。一个账号对应一个并发量（但 DeepSeek 好像最多限制二个并发）。

```toml
[server]
host = "127.0.0.1"
port = 5317

# API 访问令牌，留空则不鉴权
# [[server.api_tokens]]
# token = "sk-your-token"
# description = "开发测试"

# 邮箱和手机号二选一或都填，手机号目前好像只支持 +86
[[accounts]]
email = "user1@example.com"
mobile = ""
area_code = ""
password = "pass1"
```

这里分享一个免费的测试账号，不要发敏感信息（虽然程序每次会收尾删除会话，但是可能会遗留）。

```text
rivigol378@tatefarm.com
test12345
```

想要自己多整几个账号并发的话，可以研究一下临时邮箱（有些可能不行），然后加魔法在国际版中多注册几个账号。

推荐临时邮箱网站：[temp-mail.org](https://temp-mail.org/en/10minutemail)

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| POST | `/v1/chat/completions` | 聊天补全（支持流式与工具调用） |
| GET | `/v1/models` | 模型列表 |
| GET | `/v1/models/{id}` | 模型详情 |
| POST | `/anthropic/v1/messages` | Anthropic Messages API（支持流式与工具调用） |
| GET | `/anthropic/v1/models` | 模型列表（Anthropic 格式） |
| GET | `/anthropic/v1/models/{id}` | 模型详情（Anthropic 格式） |

## 模型映射

`config.toml` 中 `model_types`（默认 `["default", "expert"]`）自动映射：

| OpenAI 模型 ID | DeepSeek 类型 |
|----------------|--------------|
| `deepseek-default` | 快速模式 |
| `deepseek-expert` | 专家模式 |

Anthropic 兼容层使用相同的模型 ID，通过 `/anthropic/v1/messages` 调用。

### 能力开关

- **深度思考**：默认已开启。如需显式关闭，请求体中加 `"reasoning_effort": "none"`。
- **智能搜索**：默认关闭。如需开启，请求体中加 `"web_search_options": {"search_context_size": "high"}`。
- **工具调用**：按 OpenAI 标准传入 `tools` 与 `tool_choice` 即可。当模型决定调用工具时，返回的 `finish_reason` 为 `tool_calls`。

## 开发

需要 Rust 1.95.0+（见 `rust-toolchain.toml`）。

```bash
# 一键检查 (check + clippy + fmt + audit + unused deps)
just check

# 运行测试
cargo test

# 运行 HTTP 服务
just serve

# CLI 示例
just ds-core-cli
just openai-adapter-cli

# Python e2e 测试（需要服务已在 5317 端口运行）
just e2e

# 使用 e2e 专属配置启动服务
just e2e-serve
```

简要架构图：

```mermaid
flowchart TB
    %% ========== 样式定义：提高对比度，统一圆角 ==========
    classDef client fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#1e40af,rx:12,ry:12
    classDef gateway fill:#fef9c3,stroke:#ca8a04,stroke-width:3px,color:#854d0e,rx:10,ry:10
    classDef adapter fill:#fae8ff,stroke:#a21caf,stroke-width:2px,color:#701a75,rx:8,ry:8
    classDef core fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#14532d,rx:8,ry:8
    classDef external fill:#fee2e2,stroke:#dc2626,stroke-width:3px,color:#991b1b,rx:4,ry:4

    %% ========== 节点定义 ==========
    Client(["🖥️ 客户端<br/>OpenAI / Anthropic 协议"]):::client

    subgraph GW ["🌐 接入层 (axum HTTP)"]
        Server(["🔀 路由 / 鉴权"]):::gateway
    end

    subgraph PL ["📦 协议处理层 (Protocol Layer)"]
        direction LR

        subgraph AC ["🔀 Anthropic 兼容层 (anthropic_compat)"]
            A2OReq["Anthropic → OpenAI<br/>请求映射"]:::adapter
            O2AResp["OpenAI → Anthropic<br/>响应映射"]:::adapter
        end

        subgraph OA ["⚙️ OpenAI 适配层 (openai_adapter)"]
            ReqParse["请求解析"]:::adapter
            RespTrans["响应转换"]:::adapter
        end
    end

    subgraph CL ["⚡ 核心逻辑层 (ds_core)"]
        direction TB
        Pool["🔄 账号池轮转"]:::core
        Pow["⛏️ PoW 求解"]:::core
        Chat["💬 对话编排"]:::core
    end

    DeepSeek[("🔴 DeepSeek API")]:::external

    %% ========== 请求链路（实线 →） ==========
    Client -->|"OpenAI / Anthropic 协议"| Server
    Server -->|"OpenAI 流量"| ReqParse
    Server -->|"Anthropic 流量"| A2OReq
    A2OReq --> ReqParse
    ReqParse --> Pool
    Pool --> Pow
    Pow --> Chat
    Chat -->|"DeepSeek 内部协议"| DeepSeek

    %% ========== 响应链路（虚线 -.-→） ==========
    Chat -.->|"SSE 数据流"| RespTrans
    RespTrans -.->|"OpenAI 响应"| Server
    RespTrans -.->|"待转换响应"| O2AResp
    O2AResp -.->|"Anthropic 响应"| Server

    %% ========== 子图背景与边框（增强层次） ==========
    style GW fill:#fefce8,stroke:#eab308,stroke-width:2px
    style PL fill:#fafafa,stroke:#a3a3a3,stroke-width:2px
    style AC fill:#fdf4ff,stroke:#c026d3,stroke-width:1px
    style OA fill:#f5f3ff,stroke:#8b5cf6,stroke-width:1px
    style CL fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
```

数据管道：

- **OpenAI 请求**: `JSON body` → `normalize` 校验/默认值 → `tools` 提取 → `prompt` ChatML 构建 → `resolver` 模型映射 → `ChatRequest`
- **OpenAI 响应**: `DeepSeek SSE bytes` → `sse_parser` → `state` 补丁状态机 → `converter` 格式转换 → `tool_parser` XML 解析 → `StopStream` 截断 → `OpenAI SSE bytes`
- **Anthropic 请求**: `Anthropic JSON` → `to_openai_request()` → 进入 OpenAI 请求管道
- **Anthropic 响应**: OpenAI 输出 → `from_chat_completion_stream()` / `from_chat_completion_bytes()` → `Anthropic SSE/JSON`

## 许可证

[Apache License 2.0](LICENSE)

DeepSeek 官方 API 非常便宜，请大家多多支持官方服务。

本项目的初心是想体验官方网页端灰度测试的最新模型。

**严禁商用**，避免对官方服务器造成压力，否则风险自担。
