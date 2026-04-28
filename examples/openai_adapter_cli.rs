//! Interactive CLI harness for `openai_adapter`
//!
//! Usage:
//!   Interactive: `cargo run --example openai_adapter_cli`
//!   Script: `cargo run --example openai_adapter_cli -- source examples/openai_adapter_cli-script.txt`
//!
//! Commands:
//!   chat <json_file> [--raw]               - load a standard OpenAI JSON body and route via `stream`
//!   concurrent <n> <json_file> [--raw]     - concurrent chat requests
//!   models                                 - list advertised models
//!   model <id>                             - fetch metadata for one model id
//!   status                                 - inspect the underlying `ds_core` account pool
//!   source <file>                          - execute commands loaded from disk
//!   quit | exit                            - exit and shut down cleanly

use bytes::Bytes;
use ds_free_api::{Config, OpenAIAdapter, StreamResponse};
use futures::{StreamExt, future::join_all};
use std::io::{self, Read, Write};
use std::path::Path;

/// Read one line from stdin while tolerating invalid UTF-8
fn read_line_lossy() -> io::Result<String> {
    let mut buf = Vec::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    loop {
        let mut byte = [0u8; 1];
        match handle.read(&mut byte) {
            Ok(0) => break,
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

    let config = Config::load_with_args(std::env::args())?;
    println!("[Starting...]");
    let adapter = OpenAIAdapter::new(&config).await?;
    println!(
        "[Ready] commands: chat <json> [--raw] | concurrent <n> <json> [--raw] | models | model | status | source | quit"
    );

    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let line = read_line_lossy()?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if handle_line(line, &adapter).await? {
            break;
        }
    }

    println!("[Shutting down...]");
    adapter.shutdown().await;
    println!("[Stopped]");

    Ok(())
}

/// Split positional tokens from `--raw` / `-r` trailing flags
fn parse_args<'a>(parts: &'a [&'a str]) -> (Vec<&'a str>, bool) {
    let raw = parts.iter().any(|p| *p == "--raw" || *p == "-r");
    let positional: Vec<_> = parts
        .iter()
        .filter(|p| **p != "--raw" && **p != "-r")
        .copied()
        .collect();
    (positional, raw)
}

async fn handle_line(line: &str, adapter: &OpenAIAdapter) -> anyhow::Result<bool> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(false);
    }

    let cmd = parts[0];
    match cmd {
        "status" => {
            println!("[Accounts]");
            for (i, s) in adapter.account_statuses().iter().enumerate() {
                let email = if s.email.is_empty() { "-" } else { &s.email };
                let mobile = if s.mobile.is_empty() { "-" } else { &s.mobile };
                println!("  [{}] {} / {}", i + 1, email, mobile);
            }
        }

        "chat" if parts.len() >= 2 => {
            let (positional, raw) = parse_args(&parts);
            let file = positional[1];
            if !Path::new(file).exists() {
                eprintln!("[error] file not found: {}", file);
                return Ok(false);
            }
            let body = std::fs::read_to_string(file)?;
            println!(">>> request file: {}", file);
            if let Err(e) = run_chat(adapter, body.as_bytes(), raw).await {
                eprintln!("[request failed] {}", e);
            }
        }

        "concurrent" if parts.len() >= 3 => {
            let (positional, raw) = parse_args(&parts);
            let count: usize = match positional[1].parse() {
                Ok(n) if n > 0 => n,
                _ => {
                    eprintln!("[error] concurrency count must be a positive integer");
                    return Ok(false);
                }
            };
            let file = positional[2];
            if !Path::new(file).exists() {
                eprintln!("[error] file not found: {}", file);
                return Ok(false);
            }
            let body = std::fs::read_to_string(file)?;
            println!(">>> concurrent: count={}, file={}", count, file);
            run_concurrent(adapter, count, body, raw).await;
        }

        "models" => {
            let json = adapter.list_models();
            println!("{}", String::from_utf8_lossy(&json));
        }

        "model" if parts.len() == 2 => {
            if let Some(json) = adapter.get_model(parts[1]) {
                println!("{}", String::from_utf8_lossy(&json));
            } else {
                println!("null");
            }
        }

        "source" if parts.len() == 2 => {
            let file = parts[1];
            if !Path::new(file).exists() {
                eprintln!("[error] file not found: {}", file);
                return Ok(false);
            }
            println!("[Running script: {}]", file);
            let content = std::fs::read_to_string(file)?;
            for script_line in content.lines() {
                let script_line = script_line.trim();
                if script_line.is_empty() || script_line.starts_with('#') {
                    continue;
                }
                println!(">>> {}", script_line);
                if Box::pin(handle_line(script_line, adapter)).await? {
                    return Ok(true);
                }
            }
            println!("[Script finished]");
        }

        "quit" | "exit" => {
            println!("[Exiting]");
            return Ok(true);
        }

        _ => {
            println!(
                "[unknown command: {}] available: chat | concurrent | models | model | status | source | quit",
                cmd
            );
        }
    }

    Ok(false)
}

/// Return true when JSON body asks for streaming completions
fn is_stream(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false)
}

/// Dispatch a `/v1/chat/completions` body; `--raw` controls printing
async fn run_chat(adapter: &OpenAIAdapter, body: &[u8], raw: bool) -> anyhow::Result<()> {
    if is_stream(body) {
        let mut stream = adapter.chat_completions_stream(body).await?;
        print_stream(&mut stream, raw).await;
    } else {
        let json = adapter.chat_completions(body).await?;
        if raw {
            println!("{}", String::from_utf8_lossy(&json));
        } else {
            print_chat_summary(&json);
        }
    }
    Ok(())
}

/// Pretty-print the interesting fields for a buffered JSON completion
fn print_chat_summary(json: &[u8]) {
    let v: serde_json::Value = match serde_json::from_slice(json) {
        Ok(val) => val,
        Err(_) => {
            println!("{}", String::from_utf8_lossy(json));
            return;
        }
    };

    let choice = v.get("choices").and_then(|c| c.get(0));
    let message = choice.and_then(|c| c.get("message"));
    let content = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str());
    let reasoning = message
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|c| c.as_str());
    let tool_calls = message.and_then(|m| m.get("tool_calls"));
    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str());
    let usage = v.get("usage");

    let mut summary = serde_json::Map::new();
    if let Some(c) = content {
        summary.insert("content".into(), c.into());
    }
    if let Some(r) = reasoning {
        summary.insert("reasoning_content".into(), r.into());
    }
    if let Some(t) = tool_calls {
        summary.insert("tool_calls".into(), t.clone());
    }
    if let Some(f) = finish {
        summary.insert("finish_reason".into(), f.into());
    }
    if let Some(u) = usage {
        summary.insert("usage".into(), u.clone());
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&summary).unwrap_or_default()
    );
}

async fn print_stream(stream: &mut StreamResponse, raw: bool) {
    let mut stdout = io::stdout();
    while let Some(res) = stream.next().await {
        match res {
            Ok(bytes) => {
                if raw {
                    print!("{}", String::from_utf8_lossy(&bytes));
                    stdout.flush().unwrap();
                } else {
                    print_stream_chunk(&bytes);
                }
            }
            Err(e) => {
                eprintln!("\n[stream error] {}", e);
                break;
            }
        }
    }
    if !raw {
        println!();
    }
}

/// Summarize each SSE chunk for readability (non `--raw` mode)
fn print_stream_chunk(bytes: &Bytes) {
    let text = String::from_utf8_lossy(bytes);
    let json_str = text
        .strip_prefix("data: ")
        .and_then(|s| s.strip_suffix("\n\n"))
        .unwrap_or(&text);

    let v: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(val) => val,
        Err(_) => {
            print!("{}", text);
            return;
        }
    };

    let choice = v.get("choices").and_then(|c| c.get(0));
    let delta = choice.and_then(|c| c.get("delta"));
    let content = delta
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str());
    let reasoning = delta
        .and_then(|d| d.get("reasoning_content"))
        .and_then(|c| c.as_str());
    let tool_calls = delta.and_then(|d| d.get("tool_calls"));
    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str());
    let usage = v.get("usage");

    // Usage-only SSE events (empty `choices`) are handled separately.
    if choice.is_none() || usage.is_some() {
        if let Some(u) = usage {
            println!("[usage] {}", u);
            return;
        }
    }

    let mut parts = Vec::new();
    if let Some(c) = content {
        parts.push(format!("content={:?}", c));
    }
    if let Some(r) = reasoning {
        parts.push(format!("reasoning={:?}", r));
    }
    if let Some(t) = tool_calls {
        parts.push(format!(
            "tool_calls={}",
            serde_json::to_string(t).unwrap_or_default()
        ));
    }
    if let Some(f) = finish {
        parts.push(format!("finish={}", f));
    }

    if !parts.is_empty() {
        println!("[chunk] {}", parts.join(" | "));
    }
}

async fn run_concurrent(adapter: &OpenAIAdapter, count: usize, body_template: String, raw: bool) {
    let start = std::time::Instant::now();
    let body_bytes = body_template.into_bytes();
    let is_streaming = is_stream(&body_bytes);

    let futures: Vec<_> = (0..count)
        .map(|i| {
            let body = body_bytes.clone();
            async move {
                let req_start = std::time::Instant::now();
                let result = if is_streaming {
                    match adapter.chat_completions_stream(&body).await {
                        Ok(mut stream) => {
                            let mut output = String::new();
                            let mut ok = true;
                            while let Some(chunk) = stream.next().await {
                                match chunk {
                                    Ok(bytes) => {
                                        if raw {
                                            output.push_str(&String::from_utf8_lossy(&bytes));
                                        } else {
                                            let text = String::from_utf8_lossy(&bytes);
                                            let json_str = text
                                                .strip_prefix("data: ")
                                                .and_then(|s| s.strip_suffix("\n\n"))
                                                .unwrap_or(&text);
                                            if let Ok(v) =
                                                serde_json::from_str::<serde_json::Value>(json_str)
                                            {
                                                let delta = v
                                                    .get("choices")
                                                    .and_then(|c| c.get(0))
                                                    .and_then(|c| c.get("delta"));
                                                if let Some(c) = delta
                                                    .and_then(|d| d.get("content"))
                                                    .and_then(|c| c.as_str())
                                                {
                                                    output.push_str(c);
                                                }
                                                if let Some(r) = delta
                                                    .and_then(|d| d.get("reasoning_content"))
                                                    .and_then(|c| c.as_str())
                                                {
                                                    if !output.is_empty() {
                                                        output.push(' ');
                                                    }
                                                    output.push_str(r);
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("\n[request {} stream error] {}", i, e);
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            (i, ok, output, req_start.elapsed())
                        }
                        Err(e) => {
                            eprintln!("[request {} failed] {}", i, e);
                            (i, false, String::new(), req_start.elapsed())
                        }
                    }
                } else {
                    match adapter.chat_completions(&body).await {
                        Ok(json) => {
                            let output = if raw {
                                String::from_utf8_lossy(&json).to_string()
                            } else {
                                let v: serde_json::Value =
                                    serde_json::from_slice(&json).unwrap_or_default();
                                let mut parts = Vec::new();
                                if let Some(c) = v
                                    .get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("message"))
                                    .and_then(|m| m.get("content"))
                                    .and_then(|c| c.as_str())
                                {
                                    parts.push(c.to_string());
                                }
                                if let Some(r) = v
                                    .get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("message"))
                                    .and_then(|m| m.get("reasoning_content"))
                                    .and_then(|c| c.as_str())
                                {
                                    parts.push(r.to_string());
                                }
                                parts.join(" ")
                            };
                            (i, true, output, req_start.elapsed())
                        }
                        Err(e) => {
                            eprintln!("[request {} failed] {}", i, e);
                            (i, false, String::new(), req_start.elapsed())
                        }
                    }
                };
                result
            }
        })
        .collect();

    let results = join_all(futures).await;
    let total_elapsed = start.elapsed();

    println!("\n[Concurrent results]");
    let success_count = results.iter().filter(|(_, ok, _, _)| *ok).count();
    for (i, ok, output, elapsed) in results {
        let preview: String = output.chars().take(80).collect();
        let status = if ok { "ok" } else { "fail" };
        println!(
            "  [req {:2}] {} | {:>12?} | {}",
            i,
            status,
            elapsed,
            if preview.is_empty() {
                "(no output)".to_string()
            } else {
                format!("{}...", preview.replace('\n', " "))
            }
        );
    }
    println!(
        "  Total: {}/{} ok | elapsed {:?}",
        success_count, count, total_elapsed
    );
}
