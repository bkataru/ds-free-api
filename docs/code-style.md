# Code style

## Comments

### Module docs (`//!`)

- First line: what the module owns — concrete responsibility
- After a blank line: key design decisions or constraints

```rust
//! Account pool — multi-account load balancing
//!
//! 1 account = 1 session = 1 concurrency
```

### Public API docs (`///`)

- Lead with a verb: "Returns", "Creates", "Sends"
- Call out side effects: "automatically releases", "cleans up session"
- Document panic preconditions when they exist

```rust
/// Poll until an idle account is available
///
/// Returned `AccountGuard` clears the busy flag on drop
pub fn get_account(&self) -> Option<AccountGuard>
```

### Line comments (`//`)

- Explain why, not what
- Flag workarounds and external quirks

```rust
// Order matters: health_check must run before update_title,
// or an empty session triggers EMPTY_CHAT_SESSION
```

## Naming

| Kind | Style | Example |
|------|-------|---------|
| Module / file | snake_case | `ds_core`, `accounts.rs` |
| Type / struct | PascalCase | `AccountPool`, `CoreError` |
| Function / method | snake_case | `get_account()`, `compute_pow()` |
| Constant | SCREAMING_SNAKE_CASE | `ENDPOINT_USERS_LOGIN` |
| Enum variant | PascalCase | `AllAccountsFailed` |

## Error messages

- End-user wording in English
- Include context, e.g. `account {} failed to initialize`
- Never leak secrets (log at most the first 8 characters of a token)

## Logging

See `docs/logging-spec.md`

## Imports

1. Standard library (`std::`)
2. Third party (`tokio::`, `reqwest::`)
3. Crate internals (`crate::`)
4. Local `use` (`super`, `self`)

Blank line between groups.

## Tests

- `println!` is allowed inside tests to inspect parse output when a case fails
- Library code under `src/` outside `#[cfg(test)]` must not use `println!` / `eprintln!`
