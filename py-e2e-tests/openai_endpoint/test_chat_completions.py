import pytest

pytestmark = [pytest.mark.requires_server]


# =============================================================================
# Model coverage strategy
#
# Basic tests (non-stream/streaming) run on both models to ensure consistent protocol response structure.
# Extended capability tests run on deepseek-default only; expert covers basic validation.
#
# Model assignment:
#   deepseek-default  → basic + all extended capability tests
#   deepseek-expert   → basic tests
# =============================================================================

DEFAULT_MODEL = "deepseek-default"
EXPERT_MODEL = "deepseek-expert"


# =============================================================================
# Basic features (parametrized across both models)
# =============================================================================


@pytest.mark.parametrize("model", [DEFAULT_MODEL, EXPERT_MODEL], ids=["default", "expert"])
def test_non_stream_basic(client, model):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello, please answer briefly"}],
        stream=False,
    )

    assert resp.object == "chat.completion"
    assert resp.model == model
    assert len(resp.choices) == 1
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.completion_tokens > 0
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.total_tokens > 0


@pytest.mark.parametrize("model", [DEFAULT_MODEL, EXPERT_MODEL], ids=["default", "expert"])
def test_stream_basic(client, model):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello, please answer briefly"}],
        stream=True,
    )

    chunks = list(stream)
    assert chunks

    first = chunks[0]
    assert first.choices[0].delta.role == "assistant"

    content = "".join(
        c.choices[0].delta.content or "" for c in chunks if c.choices
    )
    assert content, f"Streaming response content empty, chunk count: {len(chunks)}"

    last = chunks[-1]
    assert last.choices[0].finish_reason == "stop"


# =============================================================================
# Capability switches (deepseek-default only)
# =============================================================================


def test_reasoning_effort_high(client):
    """reasoning_effort=high explicitly enables deep thinking (default)"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "1+1="}],
        reasoning_effort="high",
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_reasoning_effort_none(client):
    """reasoning_effort=none disables deep thinking"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "1+1="}],
        reasoning_effort="none",
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_web_search_enabled(client):
    """web_search_options enables smart search"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "What is the news today"}],
        web_search_options={"search_context_size": "high"},
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


# =============================================================================
# Message formats (deepseek-default only)
# =============================================================================


def test_system_message(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are a math assistant, only answer with numbers."},
            {"role": "user", "content": "2+3="},
        ],
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_developer_message(client):
    """developer role as system alternative should parse compatibly"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "developer", "content": "Answer in English."},
            {"role": "user", "content": "hello"},
        ],
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_multimodal_user(client):
    """Multimodal messages (image_url/input_audio/file) should parse without error"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image content"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "high",
                        },
                    },
                    {"type": "input_audio", "input_audio": {"data": "base64...", "format": "mp3"}},
                    {"type": "file", "file": {"filename": "report.pdf"}},
                ],
            }
        ],
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason in ("stop", "length")


def test_assistant_with_tool_calls_history(client):
    """assistant message with tool_calls history should parse normally"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": "Check Beijing weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny, 25C"},
            {"role": "user", "content": "Thanks"},
        ],
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_function_message_legacy(client):
    """Deprecated function role should parse compatibly"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": "Calculate"},
            {"role": "function", "name": "calc", "content": "42"},
        ],
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


# =============================================================================
# Stop sequences (deepseek-default only)
# =============================================================================


def test_stop_single_string(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Output the first 8 letters of the alphabet in order"}],
        stop="D",
        stream=False,
    )
    assert resp.choices[0].finish_reason == "stop"
    assert "D" not in resp.choices[0].message.content


def test_stop_multiple_strings(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Output the first 8 letters of the alphabet in order"}],
        stop=["D", "E"],
        stream=False,
    )
    assert resp.choices[0].finish_reason == "stop"


# =============================================================================
# Stream options (deepseek-default only)
# =============================================================================


def test_stream_include_usage(client):
    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = list(stream)
    assert chunks

    # At least one chunk should contain usage info
    usage_chunks = [c for c in chunks if c.usage]
    assert len(usage_chunks) >= 1

    # Last chunk with choices should have finish_reason
    finish_chunks = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finish_chunks
    assert finish_chunks[-1].choices[0].finish_reason == "stop"


# =============================================================================
# Tool Choice modes (deepseek-default only)
# =============================================================================


def test_tool_choice_required(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
        stream=False,
    )
    # required mode should trigger tool_calls
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls


def test_tool_choice_named_function(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {},
                },
            },
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=False,
    )
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_tool_choice_none_ignores_tools(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {}},
            }
        ],
        tool_choice="none",
        stream=False,
    )
    # none mode should not trigger tool_calls
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_parallel_tool_calls_false(client):
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Check weather in both Beijing and Shanghai"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        parallel_tool_calls=False,
        stream=False,
    )
    # Only require request to succeed; finish_reason may be None (empty response)
    assert resp.choices[0].finish_reason in (None, "stop", "tool_calls")


# =============================================================================
# Deprecated functions/function_call compat (deepseek-default only)
# =============================================================================


def test_functions_legacy_auto(client):
    """functions + function_call=auto should map to tools + tool_choice=auto"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Check Beijing weather"}],
        functions=[
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        function_call="auto",
        stream=False,
    )
    # After mapping should be equivalent to tools + tool_choice=auto
    assert resp.choices[0].finish_reason in ("stop", "tool_calls")
    if resp.choices[0].message.tool_calls:
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_functions_legacy_named(client):
    """function_call={'name': 'x'} should map to corresponding tool_choice"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Use the get_weather function to check Beijing weather"}],
        functions=[
            {
                "name": "get_weather",
                "description": "Get weather for a specified city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        function_call={"name": "get_weather"},
        stream=False,
    )
    assert resp.choices[0].finish_reason in ("stop", "tool_calls")
    if resp.choices[0].message.tool_calls:
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_functions_and_tools_priority(client):
    """When tools and functions coexist, prefer tools"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Check time"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {},
                },
            }
        ],
        functions=[
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            }
        ],
        tool_choice="auto",
        function_call="auto",
        stream=False,
    )
    # Should prefer get_time from tools, not overridden by functions
    if resp.choices[0].message.tool_calls:
        names = [tc.function.name for tc in resp.choices[0].message.tool_calls]
        assert "get_weather" not in names  # functions tool should not appear


# =============================================================================
# response_format fallback compat (deepseek-default only)
# =============================================================================


def test_response_format_json_object(client):
    """response_format={type: json_object} should inject JSON constraint in prompt"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Output user info including name and age"}],
        response_format={"type": "json_object"},
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_response_format_json_schema(client):
    """response_format={type: json_schema} should inject schema constraint in prompt"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Output user info"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "user_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        },
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


def test_response_format_text_no_injection(client):
    """response_format={type: text} should not inject extra constraints"""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        response_format={"type": "text"},
        stream=False,
    )
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"


# =============================================================================
# Parsed-but-ignored fields (deepseek-default only)
# =============================================================================


def test_ignored_params(client):
    """
    Pass many fields the adapter parses but does not consume; verify the request completes without error.
    These include: temperature, top_p, max_tokens, max_completion_tokens,
    frequency_penalty, presence_penalty, seed, n, metadata, store,
    user, safety_identifier, prompt_cache_key, modalities, prediction。
    """
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        max_completion_tokens=100,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        seed=42,
        n=1,
        metadata={"key": "value"},
        store=True,
        user="test-user",
        safety_identifier="safe-id",
        prompt_cache_key="cache-key",
        modalities=["text"],
        prediction={"type": "content", "content": "predicted content"},
        stream=False,
    )
    # Key assertion: request does not error and returns normal response
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content
    assert resp.choices[0].finish_reason == "stop"
