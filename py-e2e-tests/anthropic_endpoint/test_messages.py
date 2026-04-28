import pytest

pytestmark = [pytest.mark.requires_server]


# =============================================================================
# Model coverage strategy
#
# Basic tests run on both models to ensure consistent protocol response structure.
# Extended capability tests assigned per model to avoid duplication.
#
# Model assignment:
#   deepseek-default  → basic + thinking + web_search + partial message formats
#   deepseek-expert   → basic + stop_sequences + ignored_params
# =============================================================================

DEFAULT_MODEL = "deepseek-default"
EXPERT_MODEL = "deepseek-expert"


def _extract_text(msg):
    """Extract all text content from a message。"""
    return "".join(b.text for b in msg.content if b.type == "text")


# =============================================================================
# Basic features (parametrized across both models)
# =============================================================================


@pytest.mark.parametrize("model", [DEFAULT_MODEL, EXPERT_MODEL], ids=["default", "expert"])
def test_non_stream_basic(client, model):
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, please answer briefly."}],
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
        messages=[{"role": "user", "content": "Hello, please answer briefly."}],
    ) as stream:
        events = list(stream)

    assert events

    # Verify event sequence completeness
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"

    # Collect text deltas
    text_parts = []
    for event in events:
        if event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                text_parts.append(event.delta.text)

    full_text = "".join(text_parts)
    assert full_text, f"Streaming response text empty, event count: {len(events)}"

    # Verify message_delta exists
    msg_deltas = [e for e in events if e.type == "message_delta"]
    assert len(msg_deltas) == 1
    assert msg_deltas[0].delta.stop_reason in ("end_turn", "max_tokens")


# =============================================================================
# Capability switches (deepseek-default only)
# =============================================================================


def test_thinking_enabled(client):
    """thinking=enabled explicitly enables deep thinking (default)"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "1+1="}],
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_thinking_disabled(client):
    """thinking=disabled disables deep reasoning"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "1+1="}],
        thinking={"type": "disabled"},
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


def test_web_search_enabled(client):
    """web_search_options enables smart search (Anthropic protocol extension, raw HTTP)"""
    import os

    base = os.getenv("TEST_BASE_URL", "http://127.0.0.1:5317/anthropic")
    api_key = os.getenv("TEST_API_KEY", "sk-test")
    resp = client._client.post(
        f"{base}/v1/messages",
        json={
            "model": DEFAULT_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What is the news today"}],
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
# Message formats (deepseek-default only)
# =============================================================================


def test_system_message(client):
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system="You are a math assistant, only answer with numbers.",
        messages=[{"role": "user", "content": "2+3="}],
    )
    assert msg.content
    text = _extract_text(msg)
    # After system prompt, should return math-related content
    assert text, "System message test should return text content"


def test_system_as_blocks(client):
    """system as text block array should parse compatibly"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": "Answer in English."}],
        messages=[{"role": "user", "content": "hello"}],
    )
    assert msg.content
    text = _extract_text(msg)
    assert text, "System block test should return text content"


def test_multimodal_user(client):
    """Multimodal messages (image / document, etc.) should parse without errors"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image content"},
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
    """assistant messages carrying tool_use history should parse correctly"""
    msg = client.messages.create(
        model=DEFAULT_MODEL,
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
                        "content": "Sunny, 25C",
                    }
                ],
            },
            {"role": "user", "content": "Thanks"},
        ],
    )
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")


# =============================================================================
# Stop sequences (focused on deepseek-expert)
# =============================================================================


def test_stop_sequences(client):
    msg = client.messages.create(
        model=EXPERT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Output the first 8 letters of the alphabet in order"}],
        stop_sequences=["D"],
    )
    assert msg.stop_reason in ("end_turn", "stop_sequence")
    content_text = _extract_text(msg)
    # Because stop_sequence is triggered, the output should not contain "D"
    assert "D" not in content_text, f"stop_sequences should prevent 'D' from appearing. Actual output: {content_text}"


def test_stop_multiple_sequences(client):
    msg = client.messages.create(
        model=EXPERT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Output the first 8 letters of the alphabet in order"}],
        stop_sequences=["D", "E"],
    )
    assert msg.stop_reason in ("end_turn", "stop_sequence")


# =============================================================================
# Parsed-but-ignored fields (focused on deepseek-expert)
# =============================================================================


def test_ignored_params(client):
    """
    Pass many fields that the adapter parses but does not consume, and verify the request completes without errors.
    These fields include: temperature, top_p, top_k, metadata.
    """
    msg = client.messages.create(
        model=EXPERT_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        metadata={"user_id": "test-user"},
    )
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.content
    assert msg.stop_reason in ("end_turn", "max_tokens")
