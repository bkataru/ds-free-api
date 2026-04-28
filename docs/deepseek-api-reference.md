# DeepSeek endpoints

### Base URLs

- `https://chat.deepseek.com/api/v0` — all API endpoints
- `https://fe-static.deepseek.com` — WASM asset fetch

### Shared headers

- Every request: `User-Agent` required (WAF bypass)
- Authenticated calls: `Authorization: Bearer <token>`
- PoW-protected calls: `X-Ds-Pow-Response: <base64>`

### Error shapes

- **Missing fields** → HTTP 422: `{"detail":[{"loc":"body.<field>"}]}`
- **Invalid token** → HTTP 200: `{"code":40003,"msg":"Authorization Failed (invalid token)","data":null}`
- **Business error** → HTTP 200: `{"code":0,"data":{"biz_code":<N>,"biz_msg":"<msg>","biz_data":null}}`
- **Login failure** → HTTP 200: `{"code":0,"data":{"biz_code":2,"biz_msg":"PASSWORD_OR_USER_NAME_IS_WRONG"}}`

### PoW `target_path` map

| Endpoint | `target_path` |
|----------|---------------|
| completion | `/api/v0/chat/completion` |
| edit_message | `/api/v0/chat/edit_message` |
| upload_file | `/api/v0/file/upload_file` |



## 0. login

- url: https://chat.deepseek.com/api/v0/users/login
- Request headers:
  - `User-Agent`: required (WAF bypass — use a plausible browser UA)
  - `Content-Type: application/json`: optional when the HTTP stack sets it automatically
- Request payload:
```json
{
  "email": null,
  "mobile": "[phone_number]",
  "password": "<password>",
  "area_code": "+86",
  "device_id": "[any base64 or empty string; field is required]",
  "os": "web"
}
```
  - `email` / `mobile`: pick one; send `null` for the other
  - `device_id`: required field (omit → 422); value may be empty or random
  - `os`: required (omit → 422); always `"web"`
- Response:
```json
{
    "code": 0,
    "msg": "",
    "data": {
        "biz_code": 0,
        "biz_msg": "",
        "biz_data": {
            "code": 0,
            "msg": "",
            "user": {
                "id": "test",
                "token": "api-token",
                "email": "te****t@mails.tsinghua.edu.cn",
                "mobile_number": "999******99",
                "area_code": "+86",
                "status": 0,
                "id_profile": {
                    "provider": "WECHAT",
                    "id": "test",
                    "name": "test",
                    "picture": "https://static.deepseek.com/user-avatar/test",
                    "locale": "zh_CN",
                    "email": null
                },
                "id_profiles": [
                    {
                        "provider": "WECHAT",
                        "id": "test",
                        "name": "test",
                        "picture": "https://static.deepseek.com/user-avatar/test",
                        "locale": "zh_CN",
                        "email": null
                    }
                ],
                "chat": {
                    "is_muted": 0,
                    "mute_until": null
                },
                "has_legacy_chat_history": false,
                "need_birthday": false
            }
        }
    }
}
```
- Key field: `data.biz_data.user.token` (Bearer token for all later calls)
- Failure: `biz_code=2`, `biz_msg="PASSWORD_OR_USER_NAME_IS_WRONG"`



## 1. create

- url: https://chat.deepseek.com/api/v0/chat_session/create
- Request headers:
  - `Authorization: Bearer <token>`
  - `User-Agent`: required on every call (WAF bypass)
- Request payload: `{}`
- Response:
```json
{
    "code": 0,
    "msg": "",
    "data": {
        "biz_code": 0,
        "biz_msg": "",
        "biz_data": {
            "chat_session": {
                "id": "e6795fb3-272f-4782-87cf-6d6140b5bf76",
                "seq_id": 197895830,
                "agent": "chat",
                "model_type": "default",
                "title": null,
                "title_type": "WIP",
                "version": 0,
                "current_message_id": null,
                "pinned": false,
                "inserted_at": 1775732630.005,
                "updated_at": 1775732630.005
            },
            "ttl_seconds": 259200
        }
    }
}
```
- Key field: `data.biz_data.chat_session.id` (this becomes `chat_session_id` for completion)
- Note: the nested `chat_session` object is the full session snapshot



## 2. get_wasm_file

- url: https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm
- Request headers: anonymous — no auth, no `User-Agent`; plain GET works
- Request body: none (GET)
- Response: 26612 bytes, `Content-Type: application/wasm`, standard WASM (`\x00asm` magic)
- Note: the URL hash segment (`7b9ca65ddd`) may rotate — make it configurable



## 3. create_pow_challenge

- url: https://chat.deepseek.com/api/v0/chat/create_pow_challenge
- Request headers:
  - `Authorization: Bearer <token>`
  - `User-Agent`: required (omit → 429)
- Request payload: `{"target_path": "/api/v0/chat/completion"}`
- Response:
```json
{
    "code": 0,
    "msg": "",
    "data": {
        "biz_code": 0,
        "biz_msg": "",
        "biz_data": {
            "challenge": {
                "algorithm": "DeepSeekHashV1",
                "challenge": "7ffc9d19b6eed96a6fca68f8ffe30ee61035d4959e4180f187bf85b356016a96",
                "salt": "3bde54628ea8413fee87",
                "signature": "ce4678cf7a1290c2a7ac88c4195a5b8497e5fc4b0e8044e804f5a6f3af6fe462",
                "difficulty": 144000,
                "expire_at": 1775380966945,
                "expire_after": 300000,
                "target_path": "/api/v0/chat/completion"
            }
        }
    }
}
```
- Important fields: `challenge` (prefix for hash input), `salt` (suffix), `difficulty` (target threshold), `expire_at` (expiry in ms)
- `algorithm`: always `"DeepSeekHashV1"`
- `expire_after`: 300000 ms = five-minute validity window



## 4. completion

- url: https://chat.deepseek.com/api/v0/chat/completion
- Request headers:
  - `Authorization: Bearer <token>`
  - `User-Agent`: required
  - `X-Ds-Pow-Response`: required (base64 PoW answer — **recompute for each request**)
- Request payload:
```json
{
    "chat_session_id": "<id from create>",
    "parent_message_id": null,
    "model_type": "default",
    "prompt": "Hello",
    "thinking_enabled": true,
    "search_enabled": true,
    "preempt": false
}
```
- `model_type`: `"expert"` (default) | `"default"` | …
- **Note:** the current backend removed `ref_file_ids` upload wiring — handle attachments outside this path
- Response: `text/event-stream` SSE

### SSE framing

**1. `ready` — session handshake**
```
event: ready
data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}
```

**2. `update_session` — metadata tick**
```
event: update_session
data: {"updated_at":1775386361.526172}
```

**3. Incremental payload — operator grammar**

Every delta shares the same envelope: combine `"p"` (JSON pointer) with `"o"` (operator).

- **`"p"` + `"v"`** — SET (replace the field):
  ```
  data: {"p":"response/status","v":"FINISHED"}
  ```

- **`"p"` + `"o":"APPEND"` + `"v"`** — append into the field:
  ```
  data: {"p":"response/fragments/-1/content","o":"APPEND","v":","}
  ```

- **`"p"` + `"o":"SET"` + `"v"`** — explicit assignment (numbers):
  ```
  data: {"p":"response/fragments/-1/elapsed_secs","o":"SET","v":0.95}
  ```

- **`"p"` + `"o":"BATCH"` + `"v"`** — grouped updates:
  ```
  data: {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":41},{"p":"quasi_status","v":"FINISHED"}]}
  ```

- **Bare `"v"`** — continues the prior `"p"` path:
  ```
  data: {"v":"continuation"}
  ```

- **Full snapshot object** (initial tree, no `"p"` yet):
  ```
  data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","fragments":[{"id":2,"type":"RESPONSE","content":"Hello"}]}}}
  ```

### Dynamic paths under `response/`

**Content lives inside `fragments`; `-1` always means “last fragment”.**

| Path | thinking=OFF | thinking=ON | search=OFF | search=ON |
|------|:---:|:---:|:---:|:---:|
| `response/fragments/-1/content` | ✅ `type: RESPONSE` | ✅ THINK→RESPONSE | ✅ | ✅ |
| `response/fragments/-1/elapsed_secs` | ❌ | ✅ thinking seconds | - | - |
| `response/search_status` | - | - | ❌ | ✅ `SEARCHING` → `FINISHED` |
| `response/search_results` | - | - | ❌ | ✅ `{url, title, snippet}` array |
| `response/accumulated_token_usage` | ✅ | ✅ | ✅ | ✅ |
| `response/quasi_status` | ✅ | ✅ | ✅ | ✅ appears in BATCH |
| `response/status` | ✅ `WIP`→`FINISHED` | ✅ | ✅ | ✅ |

### Fragment shape

```typescript
{
  id: number,           // ordinal within the response
  type: "THINK" | "RESPONSE",
  content: string,
  elapsed_secs?: number, // present for THINK (latency)
  references: [],       // reserved (empty today)
  stage_id: number
}
```

### Reasoning vs final answer

**Rule: read `fragments[].type`.**

```
type == "THINK"     → chain-of-thought (only when thinking is on)
type == "RESPONSE"  → user-visible answer text
```

**Typical ordering (thinking=ON, search=ON):**

```
1. SEARCHING   → p=response/search_status, v="SEARCHING"
2. SEARCH      → p=response/search_results, v=[{url, title, snippet}, ...]
3. SEARCH END  → p=response/search_status, v="FINISHED"
