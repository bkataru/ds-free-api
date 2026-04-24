import pytest

pytestmark = [pytest.mark.requires_server]


# =============================================================================
# 模型覆盖策略
#
# 基础测试在两个模型上各跑一遍 —— 保证协议响应结构一致性。
# 扩展能力测试按能力分配到不同模型，避免重复。
#
# 模型分配：
#   deepseek-default  → 基础 + thinking + web_search + 部分消息格式
#   deepseek-expert   → 基础 + stop_sequences + ignored_params
# =============================================================================

DEFAULT_MODEL = "deepseek-default"
EXPERT_MODEL = "deepseek-expert"


def _extract_text(msg):
    """从消息中提取所有文本内容。"""
    return "".join(b.text for b in msg.content if b.type == "text")


# =============================================================================
# 基础功能（参数化：两个模型各跑一遍）
# =============================================================================


@pytest.mark.parametrize("model", [DEFAULT_MODEL, EXPERT_MODEL], ids=["default", "expert"])
def test_non_stream_basic(client, model):
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": "你好，请简单回答"}],
    )

    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.model == model
    assert msg.content
    assert msg.usage.input_tokens > 0
    assert msg.usage.output_tokens > 0
    assert msg.stop_reason in ("end_turn", "max_tokens")


@pytest.mark.parametrize("model", [DEFAULT_MODEL, EXPERT_MODEL], ids=["default", "expert"])
def test_stream_basic(client, model):
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": "你好，请简单回答"}],
    ) as stream:
        events = list(stream)

    assert events

    # 验证事件序列完整性
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"

    # 收集文本增量
    text_parts = []
    for event in events:
        if event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                text_parts.append(event.delta.text)

    full_text = "".join(text_parts)
    assert full_text, f"流式响应文本为空，事件数: {len(events)}"

    # 验证 message_delta 存在
    msg_deltas = [e for e in events if e.type == "message_delta"]
    assert len(msg_deltas) == 1
    assert msg_deltas[0].delta.stop_reason in ("end_turn", "max_tokens")


# =============================================================================
# 能力开关（集中在 deepseek-default）
# =============================================================================


def test_thinking_enabled(client):
    """thinking=enabled 显式开启深度思考（默认行为）"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "1+1="}],
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_thinking_disabled(client):
    """thinking=disabled 关闭深度思考"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "1+1="}],
        thinking={"type": "disabled"},
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_web_search_enabled(client):
    """web_search_options 开启智能搜索（Anthropic 协议扩展字段，直接发 raw HTTP）"""
    import os

    base = os.getenv("TEST_BASE_URL", "http://127.0.0.1:5317/anthropic")
    api_key = os.getenv("TEST_API_KEY", "sk-test")
    resp = client._client.post(
        f"{base}/v1/messages",
        json={
            "model": DEFAULT_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "今天有什么新闻"}],
            "web_search_options": {"search_context_size": "high"},
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    ).json()
    assert resp["type"] == "message"
    assert resp["role"] == "assistant"
    assert resp["model"] == DEFAULT_MODEL
    assert resp["content"]
    assert resp["stop_reason"] in ("end_turn", "max_tokens")


# =============================================================================
# 消息格式（集中在 deepseek-default）
# =============================================================================


def test_system_message(client):
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system="你是一个数学助手，只回答数字。",
        messages=[{"role": "user", "content": "2+3="}],
    )
    assert msg.content
    text = _extract_text(msg)
    # 系统提示后应返回与数学相关的内容
    assert text, "系统消息测试应返回文本内容"


def test_system_as_blocks(client):
    """system 参数为文本块数组时应兼容解析"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": "用中文回答。"}],
        messages=[{"role": "user", "content": "hello"}],
    )
    assert msg.content
    text = _extract_text(msg)
    assert text, "系统块测试应返回文本内容"


def test_multimodal_user(client):
    """多模态消息（image / document 等）应能正常解析不报错"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述一下图片内容"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                ],
            }
        ],
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_assistant_with_tool_use_history(client):
    """assistant 消息携带 tool_use 历史应能正常解析"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "查北京天气"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "get_weather",
                        "input": {"city": "北京"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "晴，25°C",
                    }
                ],
            },
            {"role": "user", "content": "谢谢"},
        ],
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


# =============================================================================
# Stop 序列（集中在 deepseek-expert）
# =============================================================================


def test_stop_sequences(client):
    msg = client.messages.create(
        model=EXPERT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "请按顺序输出字母表的前8个字母"}],
        stop_sequences=["D"],
    )
    assert msg.stop_reason in ("end_turn", "stop_sequence")
    content_text = _extract_text(msg)
    # 由于 stop_sequence 触发，输出中不应包含 "D"
    assert "D" not in content_text, f"stop_sequences 应阻止 'D' 出现，实际输出: {content_text}"


def test_stop_multiple_sequences(client):
    msg = client.messages.create(
        model=EXPERT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "请按顺序输出字母表的前8个字母"}],
        stop_sequences=["D", "E"],
    )
    assert msg.stop_reason in ("end_turn", "stop_sequence")


# =============================================================================
# 解析但忽略的字段（集中在 deepseek-expert）
# =============================================================================


def test_ignored_params(client):
    """
    传入大量适配器解析但不消费的字段，验证请求能正常完成不报错。
    这些字段包括：temperature, top_p, top_k, metadata。
    """
    msg = client.messages.create(
        model=EXPERT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        metadata={"user_id": "test-user"},
    )
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")
