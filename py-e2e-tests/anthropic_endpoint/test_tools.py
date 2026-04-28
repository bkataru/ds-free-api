import pytest

pytestmark = [pytest.mark.requires_server]

MODEL = "deepseek-default"

WEATHER_TOOL = {
    "type": "custom",
    "name": "get_weather",
    "description": "Get weather for a specified city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
}


def _extract_tool_use_blocks(msg):
    """Extract all tool_use blocks from a message."""
    return [b for b in msg.content if b.type == "tool_use"]


def _extract_text(msg):
    """Extract all text content from a message."""
    return "".join(b.text for b in msg.content if b.type == "text")


# =============================================================================
# Basic tool calling
# =============================================================================


def test_tool_call_forced(client):
    """Forced mode must trigger tool_use with correct parameters."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "any"},
    )

    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.model == MODEL
    assert msg.stop_reason == "tool_use"

    tool_blocks = _extract_tool_use_blocks(msg)
    assert len(tool_blocks) == 1, f"Expected 1 tool_use block, got {len(tool_blocks)}"

    tb = tool_blocks[0]
    assert tb.name == "get_weather"
    assert tb.id.startswith("toolu_")
    assert isinstance(tb.input, dict)
    assert "city" in tb.input
    assert tb.input["city"] == "Beijing"


def test_tool_call_stream_event_sequence(client):
    """Streaming tool call must output complete event sequence: start -> input_json_delta -> stop."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "any"},
    ) as stream:
        events = list(stream)

    assert events

    # Verify message_start
    assert events[0].type == "message_start"
    assert events[0].message.id.startswith("msg_")
    assert events[0].message.role == "assistant"

    # Collect all tool_use related events
    tool_starts = [e for e in events if e.type == "content_block_start" and e.content_block.type == "tool_use"]
    tool_deltas = [e for e in events if e.type == "content_block_delta" and hasattr(e.delta, "partial_json")]
    tool_stops = [e for e in events if e.type == "content_block_stop"]

    assert len(tool_starts) == 1, f"Expected 1 tool_use start, got {len(tool_starts)}"
    assert len(tool_deltas) == 1, f"Expected 1 input_json_delta, got {len(tool_deltas)}"
    # May have thinking block stop, so total stops >= 1
    assert len(tool_stops) >= 1, f"Expected at least 1 content_block_stop, got {len(tool_stops)}"

    # Verify start event
    start = tool_starts[0]
    assert start.content_block.name == "get_weather"
    assert start.content_block.id.startswith("toolu_")
    assert start.content_block.input == {}, "input should be empty object at start"

    # Verify delta event
    delta = tool_deltas[0]
    assert delta.delta.partial_json
    parsed = pytest.importorskip("json").loads(delta.delta.partial_json)
    assert isinstance(parsed, dict)
    assert "city" in parsed

    # Verify message_delta
    msg_deltas = [e for e in events if e.type == "message_delta"]
    assert len(msg_deltas) == 1
    assert msg_deltas[0].delta.stop_reason == "tool_use"

    # Verify message_stop
    assert events[-1].type == "message_stop"


def test_tool_call_no_force_falls_back_to_text(client):
    """Without forcing, the model may answer directly."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "How is the weather today?"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "auto"},
    )

    assert msg.type == "message"
    assert msg.stop_reason in ("end_turn", "tool_use")

    tool_blocks = _extract_tool_use_blocks(msg)
    text = _extract_text(msg)

    # Must include at least one of tool_use or text
    assert tool_blocks or text, "Response must include tool_use or text"


# =============================================================================
# Tool Choice mode
# =============================================================================


def test_tool_choice_required_any(client):
    """tool_choice=any maps to required and must trigger tool_use."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "any"},
    )
    tool_blocks = _extract_tool_use_blocks(msg)
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "get_weather"
    assert msg.stop_reason == "tool_use"


def test_tool_choice_none_ignores_tools(client):
    """tool_choice=none should not trigger tool_use."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "none"},
    )
    tool_blocks = _extract_tool_use_blocks(msg)
    assert not tool_blocks
    text = _extract_text(msg)
    assert text
    assert msg.stop_reason == "end_turn"


def test_disable_parallel_tool_use(client):
    """disable_parallel_tool_use maps to parallel_tool_calls=false."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Check weather in both Beijing and Shanghai"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "tool_use")


# =============================================================================
# Tool call round-trip
# =============================================================================


def test_tool_use_followed_by_tool_result(client):
    """Full tool call/result round-trip should work."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Check Beijing weather"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "get_weather",
                        "input": {"city": "Beijing"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "Sunny, 25°C",
                    }
                ],
            },
        ],
        tools=[WEATHER_TOOL],
    )
    assert msg.type == "message"
    assert msg.content
