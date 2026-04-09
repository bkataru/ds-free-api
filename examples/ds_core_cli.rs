//! ds-core 交互式 CLI 测试工具
//!
//! 使用方式:
//!   交互模式: cargo run --example ds_core_cli
//!   脚本模式: cargo run --example ds_core_cli -- source test_ds_core_script.txt
//!
//! 命令:
//!   status                              - 查看所有账号状态
//!   chat <prompt> [thinking] [search]   - 发送对话 (t=thinking, s=search, 默认 false)
//!   concurrent <n> <prompt> [t] [s]     - 并发发送 n 个请求
//!   source <file>                       - 从文件读取命令执行
//!   quit | exit                         - 退出并清理

use ai_free_api::{ChatRequest, Config, DeepSeekCore};
use bytes::Bytes;
use futures::{StreamExt, future::join_all};
use std::io::{self, Read, Write};
use std::path::Path;

/// 读取一行输入，允许无效的 UTF-8
fn read_line_lossy() -> io::Result<String> {
    let mut buf = Vec::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    loop {
        let mut byte = [0u8; 1];
        match handle.read(&mut byte) {
            Ok(0) => break, // EOF
            Ok(_) => {
                if byte[0] == b'\n' {
                    break;
                }
                if byte[0] != b'\r' {
                    buf.push(byte[0]);
                }
            }
            Err(e) => return Err(e),
        }
    }

    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::new().default_filter_or("info")).init();

    let config = Config::load("config.toml")?;
    println!("[初始化中...]");
    let core = DeepSeekCore::new(&config).await?;
    println!("[就绪] 命令: status | chat | concurrent <n> | source | quit");

    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let line = read_line_lossy()?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let cmd = parts[0];

        match cmd {
            "status" => {
                println!("[账号状态]");
                for (i, s) in core.account_statuses().iter().enumerate() {
                    let email = if s.email.is_empty() { "-" } else { &s.email };
                    let mobile = if s.mobile.is_empty() { "-" } else { &s.mobile };
                    println!("  [{}] {} / {}", i + 1, email, mobile);
                }
            }

            "chat" if parts.len() >= 2 => {
                let rest = line.strip_prefix("chat").unwrap().trim();
                let (prompt, thinking, search) = parse_chat_args(rest);

                println!(
                    "[请求] prompt={:?}, thinking={}, search={}",
                    prompt.chars().take(50).collect::<String>(),
                    thinking,
                    search
                );

                match core
                    .v0_chat(ChatRequest {
                        prompt,
                        thinking_enabled: thinking,
                        search_enabled: search,
                    })
                    .await
                {
                    Ok(mut stream) => {
                        println!("[响应流开始]");
                        while let Some(chunk) = stream.next().await {
                            let chunk: Result<Bytes, ai_free_api::CoreError> = chunk;
                            match chunk {
                                Ok(bytes) => {
                                    print!("{}", String::from_utf8_lossy(&bytes));
                                    stdout.flush()?;
                                }
                                Err(e) => {
                                    eprintln!("\n[流错误] {}", e);
                                    break;
                                }
                            }
                        }
                        println!("\n[响应流结束]");
                    }
                    Err(e) => {
                        eprintln!("[请求失败] {}", e);
                    }
                }
            }

            "concurrent" if parts.len() >= 3 => {
                let count: usize = match parts[1].parse() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        eprintln!("[错误] 并发数必须是正整数");
                        continue;
                    }
                };

                let rest = line.strip_prefix("concurrent").unwrap().trim();
                let rest = rest.strip_prefix(parts[1]).unwrap().trim();
                let (prompt, thinking, search) = parse_chat_args(rest);

                println!(
                    "[并发请求] count={}, prompt={:?}, thinking={}, search={}",
                    count,
                    prompt.chars().take(50).collect::<String>(),
                    thinking,
                    search
                );

                run_concurrent(&core, count, prompt, thinking, search).await;
            }

            "source" if parts.len() == 2 => {
                let file = parts[1];
                if !Path::new(file).exists() {
                    eprintln!("[错误] 文件不存在: {}", file);
                    continue;
                }
                println!("[执行脚本: {}]", file);
                let content = std::fs::read_to_string(file)?;
                for script_line in content.lines() {
                    let script_line = script_line.trim();
                    if script_line.is_empty() || script_line.starts_with('#') {
                        continue;
                    }
                    println!(">>> {}", script_line);
                    if handle_line(script_line, &core).await? {
                        break;
                    }
                }
                println!("[脚本执行完毕]");
            }

            "quit" | "exit" => {
                println!("[退出]");
                break;
            }

            _ => {
                println!(
                    "[未知命令: {}] 可用: status | chat | concurrent | source | quit",
                    cmd
                );
            }
        }
    }

    println!("[清理中...]");
    core.shutdown().await;
    println!("[已关闭]");

    Ok(())
}

/// 运行并发请求
async fn run_concurrent(
    core: &DeepSeekCore,
    count: usize,
    prompt: String,
    thinking: bool,
    search: bool,
) {
    let start = std::time::Instant::now();

    // 创建所有请求
    let futures: Vec<_> = (0..count)
        .map(|i| {
            let prompt = prompt.clone();
            async move {
                let req_start = std::time::Instant::now();
                match core
                    .v0_chat(ChatRequest {
                        prompt,
                        thinking_enabled: thinking,
                        search_enabled: search,
                    })
                    .await
                {
                    Ok(mut stream) => {
                        let mut output = String::new();
                        while let Some(chunk) = stream.next().await {
                            match chunk {
                                Ok(bytes) => {
                                    output.push_str(&String::from_utf8_lossy(&bytes));
                                }
                                Err(e) => {
                                    eprintln!("\n[请求{} 流错误] {}", i, e);
                                    break;
                                }
                            }
                        }
                        let elapsed = req_start.elapsed();
                        (i, true, output, elapsed)
                    }
                    Err(e) => {
                        let elapsed = req_start.elapsed();
                        eprintln!("[请求{} 失败] {}", i, e);
                        (i, false, String::new(), elapsed)
                    }
                }
            }
        })
        .collect();

    // 等待所有请求完成
    let results = join_all(futures).await;
    let total_elapsed = start.elapsed();

    // 打印结果摘要
    println!("\n[并发结果]");
    let success_count = results.iter().filter(|(_, ok, _, _)| *ok).count();
    for (i, ok, output, elapsed) in results {
        let preview: String = output.chars().take(100).collect();
        let status = if ok { "成功" } else { "失败" };
        println!(
            "  [请求{:2}] {} | {:>12?} | {}",
            i,
            status,
            elapsed,
            if preview.is_empty() {
                "(无输出)".to_string()
            } else {
                format!("{}...", preview.replace('\n', " "))
            }
        );
    }
    println!(
        "  总计: {}/{} 成功 | 总耗时 {:?}",
        success_count, count, total_elapsed
    );
}

/// 解析 chat/concurrent 命令参数
fn parse_chat_args(rest: &str) -> (String, bool, bool) {
    let tokens: Vec<&str> = rest.split_whitespace().collect();

    let mut thinking = false;
    let mut search = false;
    let mut prompt_end = tokens.len();

    if let Some(last) = tokens.last() {
        if is_bool(last) && tokens.len() >= 2 {
            search = parse_bool(last);
            prompt_end -= 1;

            if let Some(second_last) = tokens.get(tokens.len() - 2) {
                if is_bool(second_last) {
                    thinking = parse_bool(second_last);
                    prompt_end -= 1;
                }
            }
        } else if is_bool(last) {
            thinking = parse_bool(last);
            prompt_end -= 1;
        }
    }

    let prompt = tokens[..prompt_end].join(" ");
    (prompt, thinking, search)
}

fn is_bool(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "true" | "false" | "t" | "f" | "1" | "0"
    )
}

fn parse_bool(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "true" | "t" | "1")
}

/// 处理单行命令
async fn handle_line(line: &str, core: &DeepSeekCore) -> anyhow::Result<bool> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(false);
    }

    match parts[0] {
        "status" => {
            println!("[账号状态]");
            for (i, s) in core.account_statuses().iter().enumerate() {
                let email = if s.email.is_empty() { "-" } else { &s.email };
                let mobile = if s.mobile.is_empty() { "-" } else { &s.mobile };
                println!("  [{}] {} / {}", i + 1, email, mobile);
            }
        }

        "chat" if parts.len() >= 2 => {
            let rest = line.strip_prefix("chat").unwrap().trim();
            let (prompt, thinking, search) = parse_chat_args(rest);

            println!(
                "[请求] prompt={:?}, thinking={}, search={}",
                prompt.chars().take(50).collect::<String>(),
                thinking,
                search
            );

            match core
                .v0_chat(ChatRequest {
                    prompt,
                    thinking_enabled: thinking,
                    search_enabled: search,
                })
                .await
            {
                Ok(mut stream) => {
                    println!("[响应流开始]");
                    let mut stdout = io::stdout();
                    while let Some(chunk) = stream.next().await {
                        let chunk: Result<Bytes, _> = chunk;
                        match chunk {
                            Ok(bytes) => {
                                print!("{}", String::from_utf8_lossy(&bytes));
                                stdout.flush()?;
                            }
                            Err(e) => {
                                eprintln!("\n[流错误] {}", e);
                                break;
                            }
                        }
                    }
                    println!("\n[响应流结束]");
                }
                Err(e) => {
                    eprintln!("[请求失败] {}", e);
                }
            }
        }

        "concurrent" if parts.len() >= 3 => {
            let count: usize = match parts[1].parse() {
                Ok(n) if n > 0 => n,
                _ => {
                    eprintln!("[错误] 并发数必须是正整数");
                    return Ok(false);
                }
            };

            let rest = line.strip_prefix("concurrent").unwrap().trim();
            let rest = rest.strip_prefix(parts[1]).unwrap().trim();
            let (prompt, thinking, search) = parse_chat_args(rest);

            run_concurrent(core, count, prompt, thinking, search).await;
        }

        "quit" | "exit" => return Ok(true),

        _ => println!("[未知命令: {}]", parts[0]),
    }

    Ok(false)
}
