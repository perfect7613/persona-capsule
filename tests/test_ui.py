from persona_capsule.ingestion import build_ingestion_draft
from persona_capsule.ui import _recover_ingestion_draft

SAMPLE = "\n".join(
    [
        "You: Let us test the smallest useful version before expanding the architecture.",
        "You: First isolate the risky assumption, then choose one measurable result.",
        "You: Thanks for documenting the tradeoff and the reversible next action.",
        "You: I am not convinced yet; what evidence would change the decision?",
        "You: Show one concrete example so we can discuss behavior instead of abstraction.",
        "You: The current plan has too many moving pieces for this first experiment.",
        "You: We need enough signal to choose the next useful action with confidence.",
        "You: Summarize the decision, the reason, and the owner of the next step.",
    ]
)


def test_recover_ingestion_draft_uses_state_when_available() -> None:
    draft = build_ingestion_draft(SAMPLE, "You", True)

    recovered = _recover_ingestion_draft(draft, "", "", False)

    assert recovered is draft


def test_recover_ingestion_draft_rebuilds_dropped_gradio_state() -> None:
    expected = build_ingestion_draft(SAMPLE, "You", True)

    recovered = _recover_ingestion_draft(None, SAMPLE, "You", True)

    assert recovered == expected
