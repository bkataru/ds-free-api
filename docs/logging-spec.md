# 日志系统规范

## 原则

1. **库代码零输出**：`ds_core/` 等库模块只使用 `log` crate，绝不直接打印到 stdout/stderr
2. **调用方控制权**：日志级别、格式、输出目标由应用层（main.rs / examples）决定
3. **结构化 target**：通过 target 路径实现模块级过滤

## 日志级别

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| `ERROR` | 需要人工介入的致命错误 | 所有账号初始化失败、PoW 模块崩溃、配置错误 |
| `WARN` | 降级但可继续的异常 | 单个账号初始化失败（还有其他账号可用）、session 清理失败 |
| `INFO` | 关键生命周期事件 | 账号初始化成功、服务启动/关闭、会话创建完成 |
| `DEBUG` | 调试信息 | HTTP 请求/响应摘要、PoW 计算耗时、SSE 事件类型 |
| `TRACE` | 最细粒度数据 | SSE 原始字节内容、完整 HTTP body |

## Target 规范

格式：`crate::module` 或 `crate::module::submodule`

| 模块 | Target | 说明 |
|------|--------|------|
| `ds_core::accounts` | `ds_core::accounts` | 账号池生命周期、健康检查 |
| `ds_core::client` | `ds_core::client` | HTTP 请求/响应、API 调用 |
| `ds_core::completions` | `ds_core::completions` | 对话编排、SSE 流处理 |
| `ds_core::pow` | `ds_core::pow` | PoW 计算、WASM 加载 |
| `adapter` | `adapter` | OpenAI 协议适配层 |

## 代码规范

### 库代码（ds_core/）

```rust
use log::{info, debug, warn, error};

// INFO: 关键生命周期
info!(target: "ds_core::accounts", "账号 {} 初始化成功", mobile);

// WARN: 单个失败可降级
warn!(target: "ds_core::accounts", "账号 {} 初始化失败: {}", mobile, e);

// DEBUG: HTTP 调试信息
debug!(target: "ds_core::client", "PoW challenge: alg={} difficulty={}", alg, diff);

// ERROR: 致命错误
error!(target: "ds_core::accounts", "所有账号初始化失败");
```

### 应用层（examples/ / main.rs）

```rust
fn main() {
    // 默认 info 级别，可通过 RUST_LOG 覆盖
    env_logger::Builder::from_env(
        env_logger::Env::new().default_filter_or("info")
    ).init();
}
```

## 运行时控制

```bash
# 默认级别（info）
cargo run --example ds_core_cli

# 调试模式 - 查看所有 debug 日志
RUST_LOG=debug cargo run --example ds_core_cli

# 模块级过滤 - 只看 accounts 的 debug
RUST_LOG=ds_core::accounts=debug cargo run --example ds_core_cli

# 多级组合 - accounts 用 debug，其他用 warn
RUST_LOG=ds_core::accounts=debug,ds_core::client=warn,info cargo run --example ds_core_cli

# 完全静默（仅错误）
RUST_LOG=error cargo run --example ds_core_cli

# 输出到文件
RUST_LOG=debug cargo run --example ds_core_cli 2> ds_core.log
```

## 禁止事项

- ❌ 库代码中直接使用 `println!` / `eprintln!`
- ❌ 使用无 target 的日志宏（如 `log::info!` 不加 target）
- ❌ 在日志中打印敏感信息（token、password）
- ❌ 高频 TRACE 日志（如每个 SSE 字节）默认开启

## 依赖配置

**Cargo.toml**
```toml
[dependencies]
log = "0.4"

[dev-dependencies]
env_logger = { version = "0.11", default-features = false, features = ["auto-color"] }
```

注：`auto-color` 特性在终端中自动添加颜色，在非 TTY 环境自动禁用。
