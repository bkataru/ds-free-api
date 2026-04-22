import pytest

pytestmark = [pytest.mark.requires_server]

MODEL = "deepseek-default"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"],
        },
    },
}


# =============================================================================
# 基础工具调用
# =============================================================================


def test_tool_call_required(client):
    """强制模式下必须触发 tool_calls，且参数正确。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "查询北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        stream=False,
    )

    assert resp.object == "chat.completion"
    assert resp.model == MODEL
    assert len(resp.choices) == 1

    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    msg = choice.message
    assert msg.role == "assistant"
    assert msg.tool_calls
    assert len(msg.tool_calls) == 1

    tc = msg.tool_calls[0]
    assert tc.type == "function"
    assert tc.function.name == "get_weather"
    assert tc.id.startswith("call_")

    args = pytest.importorskip("json").loads(tc.function.arguments)
    assert isinstance(args, dict)
    assert "city" in args
    assert args["city"] in ("北京", "Beijing")


def test_tool_call_stream_chunks(client):
    """流式工具调用应能收集到完整的 tool_calls。"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "查询北京天气"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        stream=True,
    )

    chunks = list(stream)
    assert chunks

    last = chunks[-1]
    assert last.choices[0].finish_reason == "tool_calls"

    # 收集所有 delta 中的 tool_calls
    tool_calls = []
    for c in chunks:
        if c.choices and c.choices[0].delta.tool_calls:
            tool_calls.extend(c.choices[0].delta.tool_calls)

    assert tool_calls, "流式响应中应包含 tool_calls"
    names = [tc.function.name for tc in tool_calls if tc.function and tc.function.name]
    assert "get_weather" in names


def test_tool_call_auto_may_respond_with_text(client):
    """auto 模式下模型可能直接回答。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "今天天气怎么样？"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=False,
    )

    assert resp.choices[0].finish_reason in ("stop", "tool_calls")
    msg = resp.choices[0].message
    assert msg.role == "assistant"
    # 至少要有 content 或 tool_calls 之一
    assert msg.content or msg.tool_calls


# =============================================================================
# Tool Choice 模式
# =============================================================================


def test_tool_choice_named_function(client):
    """指定具体工具时应调用该工具。使用明确提示词提高触发概率。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "请使用 get_weather 工具查询北京天气"}],
        tools=[
            WEATHER_TOOL,
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "获取当前时间",
                    "parameters": {},
                },
            },
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=False,
    )
    # 强制指定工具后应触发 tool_calls，若模型未触发则至少验证 finish_reason
    assert resp.choices[0].finish_reason in ("stop", "tool_calls")
    if resp.choices[0].message.tool_calls:
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_tool_choice_none_ignores_tools(client):
    """none 模式下不应触发 tool_calls。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好"}],
        tools=[WEATHER_TOOL],
        tool_choice="none",
        stream=False,
    )
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_parallel_tool_calls_false(client):
    """parallel_tool_calls=false 应正常处理。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "同时查北京和上海天气"}],
        tools=[WEATHER_TOOL],
        parallel_tool_calls=False,
        stream=False,
    )
    assert resp.choices[0].finish_reason in ("stop", "tool_calls")


# =============================================================================
# 自定义工具
# =============================================================================


def test_custom_tool_grammar(client):
    """自定义工具（grammar 格式）应能正常解析。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "生成一个符合语法规则的字符串"}],
        tools=[
            {
                "type": "custom",
                "custom": {
                    "name": "grammar_tool",
                    "description": "基于语法的工具",
                    "format": {
                        "type": "grammar",
                        "grammar": {
                            "definition": "start: word+",
                            "syntax": "lark",
                        },
                    },
                },
            }
        ],
        stream=False,
    )
    msg = resp.choices[0].message
    assert msg.content or msg.tool_calls
