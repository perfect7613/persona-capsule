from persona_capsule.demo import demo_reply


def test_demo_reply_is_deterministic() -> None:
    first = demo_reply("Help me explain this tradeoff.", "Balanced")
    second = demo_reply("Help me explain this tradeoff.", "Balanced")

    assert first == second
    assert "live MiniCPM steering arrives in Slice 4" in first


def test_demo_reply_requests_context_for_empty_input() -> None:
    assert "Give the capsule a situation" in demo_reply("  ", "Subtle")
