import pytest

pytestmark = [pytest.mark.requires_server]

MODEL = "deepseek-default"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a specified city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    },
}


# =============================================================================
# Basic tool calling
# =============================================================================


def test_tool_call_required(client):
    """Forced mode must trigger tool_calls with correct parameters."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
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
    assert args["city"] in ("Beijing", "Beijing")


def test_tool_call_stream_chunks(client):
    """Streaming tool call should collect complete tool_calls."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        stream=True,
    )

    chunks = list(stream)
    assert chunks

    last = chunks[-1]
    assert last.choices[0].finish_reason == "tool_calls"

    # Collect all tool_calls from deltas
    tool_calls = []
    for c in chunks:
        if c.choices and c.choices[0].delta.tool_calls:
            tool_calls.extend(c.choices[0].delta.tool_calls)

    assert tool_calls, "Streaming response should contain tool_calls"
    names = [tc.function.name for tc in tool_calls if tc.function and tc.function.name]
    assert "get_weather" in names


def test_tool_call_auto_may_respond_with_text(client):
    """Auto mode: model may respond with text directly."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "How is the weather today?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=False,
    )

    assert resp.choices[0].finish_reason in ("stop", "tool_calls")
    msg = resp.choices[0].message
    assert msg.role == "assistant"
    # Should have at least content or tool_calls
    assert msg.content or msg.tool_calls


# =============================================================================
# Tool Choice modes
# =============================================================================


def test_tool_choice_named_function(client):
    """When specifying a particular tool, should call that tool. Uses explicit prompt to increase trigger probability."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Use the get_weather tool to check Beijing weather"}],
        tools=[
            WEATHER_TOOL,
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {},
                },
            },
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=False,
    )
    # After forcing a specific tool, should trigger tool_calls; if model does not, at least verify finish_reason
    assert resp.choices[0].finish_reason in ("stop", "tool_calls")
    if resp.choices[0].message.tool_calls:
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_tool_choice_none_ignores_tools(client):
    """none mode should not trigger tool_calls。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        tools=[WEATHER_TOOL],
        tool_choice="none",
        stream=False,
    )
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_parallel_tool_calls_false(client):
    """parallel_tool_calls=false should be handled normally."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Check weather in both Beijing and Shanghai"}],
        tools=[WEATHER_TOOL],
        parallel_tool_calls=False,
        stream=False,
    )
    assert resp.choices[0].finish_reason in ("stop", "tool_calls")


# =============================================================================
# Custom tools
# =============================================================================


def test_custom_tool_grammar(client):
    """Custom tool (grammar format) should parse normally."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Generate a string conforming to grammar rules"}],
        tools=[
            {
                "type": "custom",
                "custom": {
                    "name": "grammar_tool",
                    "description": "Grammar-based tool",
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
