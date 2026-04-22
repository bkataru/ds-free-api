import pytest

pytestmark = [pytest.mark.requires_server]

MODEL = "deepseek-default"

WEATHER_TOOL = {
    "type": "custom",
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"],
    },
}


def _extract_tool_use_blocks(msg):
    """从消息中提取所有 tool_use 块。"""
    return [b for b in msg.content if b.type == "tool_use"]


def _extract_text(msg):
    """从消息中提取所有文本内容。"""
    return "".join(b.text for b in msg.content if b.type == "text")


# =============================================================================
# 基础工具调用
# =============================================================================


def test_tool_call_forced(client):
    """强制模式下必须触发 tool_use，且参数正确。"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "查询北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "any"},
    )

    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.model == MODEL
    assert msg.stop_reason == "tool_use"

    tool_blocks = _extract_tool_use_blocks(msg)
    assert len(tool_blocks) == 1, f"期望 1 个 tool_use 块，实际 {len(tool_blocks)}"

    tb = tool_blocks[0]
    assert tb.name == "get_weather"
    assert tb.id.startswith("toolu_")
    assert isinstance(tb.input, dict)
    assert "city" in tb.input
    assert tb.input["city"] in ("北京", "Beijing")


def test_tool_call_stream_event_sequence(client):
    """流式工具调用必须输出完整事件序列：start -> input_json_delta -> stop。"""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "查询北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "any"},
    ) as stream:
        events = list(stream)

    assert events

    # 验证 message_start
    assert events[0].type == "message_start"
    assert events[0].message.id.startswith("msg_")
    assert events[0].message.role == "assistant"

    # 收集所有 tool_use 相关事件
    tool_starts = [e for e in events if e.type == "content_block_start" and e.content_block.type == "tool_use"]
    tool_deltas = [e for e in events if e.type == "content_block_delta" and hasattr(e.delta, "partial_json")]
    tool_stops = [e for e in events if e.type == "content_block_stop"]

    assert len(tool_starts) == 1, f"期望 1 个 tool_use start，实际 {len(tool_starts)}"
    assert len(tool_deltas) == 1, f"期望 1 个 input_json_delta，实际 {len(tool_deltas)}"
    # 可能有 thinking block 的 stop，所以总 stop 数 >= 1
    assert len(tool_stops) >= 1, f"期望至少 1 个 content_block_stop，实际 {len(tool_stops)}"

    # 验证 start 事件
    start = tool_starts[0]
    assert start.content_block.name == "get_weather"
    assert start.content_block.id.startswith("toolu_")
    assert start.content_block.input == {}, "start 时 input 应为空对象"

    # 验证 delta 事件
    delta = tool_deltas[0]
    assert delta.delta.partial_json
    parsed = pytest.importorskip("json").loads(delta.delta.partial_json)
    assert isinstance(parsed, dict)
    assert "city" in parsed

    # 验证 message_delta
    msg_deltas = [e for e in events if e.type == "message_delta"]
    assert len(msg_deltas) == 1
    assert msg_deltas[0].delta.stop_reason == "tool_use"

    # 验证 message_stop
    assert events[-1].type == "message_stop"


def test_tool_call_no_force_falls_back_to_text(client):
    """非强制模式下，模型可能直接回答。"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "今天天气怎么样？"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "auto"},
    )

    assert msg.type == "message"
    assert msg.stop_reason in ("end_turn", "tool_use")

    tool_blocks = _extract_tool_use_blocks(msg)
    text = _extract_text(msg)

    # 至少要有 tool_use 或 text 之一
    assert tool_blocks or text, "响应必须包含 tool_use 或 text"


# =============================================================================
# Tool Choice 模式
# =============================================================================


def test_tool_choice_required_any(client):
    """tool_choice=any 映射为 required，应触发 tool_use。"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "查北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "any"},
    )
    tool_blocks = _extract_tool_use_blocks(msg)
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "get_weather"
    assert msg.stop_reason == "tool_use"


def test_tool_choice_none_ignores_tools(client):
    """tool_choice=none 不应触发 tool_use。"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "你好"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "none"},
    )
    tool_blocks = _extract_tool_use_blocks(msg)
    assert not tool_blocks
    text = _extract_text(msg)
    assert text
    assert msg.stop_reason == "end_turn"


def test_disable_parallel_tool_use(client):
    """disable_parallel_tool_use 映射为 parallel_tool_calls=false。"""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "同时查北京和上海天气"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "tool_use")


# =============================================================================
# 工具调用历史回环
# =============================================================================


def test_tool_use_followed_by_tool_result(client):
    """完整的工具调用-结果回环应正常工作。"""
    msg = client.messages.create(
        model=MODEL,
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
        ],
        tools=[WEATHER_TOOL],
    )
    assert msg.type == "message"
    assert msg.content
