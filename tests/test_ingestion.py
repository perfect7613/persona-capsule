import pytest

from persona_capsule.ingestion import (
    IngestionError,
    approve_draft,
    build_ingestion_draft,
    parse_messages,
    propose_exemplar_pairs,
    redact_messages,
    select_speaker,
    validate_sample,
)
from persona_capsule.repository import CapsuleRecord, InMemoryCapsuleRepository

SAMPLE = """
Alex: I think we should start with the smallest useful version, then measure it.
Sam: Fine by me.
Alex: Thanks — that makes the tradeoff much clearer!
Alex: I’m not convinced yet; what evidence would change our minds?
Alex: First, isolate the risky assumption. Next, test it without rebuilding everything.
Alex: Please send the draft to alex@example.com when it is ready.
Alex: The link is https://example.com/private and ticket ID AB-12345.
Alex: Honestly, this is close. I’d tighten the opening and make the action explicit.
Alex: We can disagree on the mechanism while agreeing on the outcome.
Alex: Could we name the failure mode before choosing the architecture?
Alex: The current version works, but it hides too much state for me to trust it.
Alex: Let’s write down the decision, owner, and next checkpoint.
""".strip()


def test_parse_select_validate_and_redact_messages() -> None:
    parsed = parse_messages(SAMPLE)
    selected = select_speaker(parsed, "Alex")

    validate_sample(selected)
    cleaned, redactions = redact_messages(selected)

    assert len(parsed) == 12
    assert len(selected) == 11
    assert "[EMAIL_1]" in cleaned[4].text
    assert "[URL_1]" in cleaned[5].text
    assert "[IDENTIFIER_1]" in cleaned[5].text
    assert {redaction.kind for redaction in redactions} == {
        "EMAIL",
        "IDENTIFIER",
        "URL",
    }


def test_validation_explains_insufficient_and_repetitive_samples() -> None:
    with pytest.raises(IngestionError, match="at least 8"):
        build_ingestion_draft("You: one\nYou: two", "You", True)

    repeated = "\n".join(["You: This is the same copied message."] * 10)
    with pytest.raises(IngestionError, match="too repetitive"):
        build_ingestion_draft(repeated, "You", True)


def test_consent_is_required() -> None:
    with pytest.raises(IngestionError, match="own these messages"):
        build_ingestion_draft(SAMPLE, "Alex", False)


def test_profile_schema_and_exemplar_approval_are_deterministic() -> None:
    first = build_ingestion_draft(SAMPLE, "Alex", True)
    second = build_ingestion_draft(SAMPLE, "Alex", True)

    assert first.profile == second.profile
    assert len(first.proposed_pairs) == 4
    assert all(pair.positive and pair.neutral for pair in first.proposed_pairs)
    assert all(pair.positive.casefold() != pair.neutral.casefold() for pair in first.proposed_pairs)
    assert 0 <= first.profile.uncertainty <= 1
    assert set(first.profile.dimensions.as_dict()) == {
        "openness",
        "conscientiousness",
        "expressiveness",
        "agreeableness",
        "emotional_range",
        "directness",
        "formality",
    }

    approved = approve_draft(
        first,
        {first.proposed_pairs[0].pair_hash, first.proposed_pairs[2].pair_hash},
        {"directness": 91, "formality": 17},
    )

    assert len(approved.exemplar_pairs) == 2
    assert approved.profile.dimensions.directness == 91
    assert approved.profile.dimensions.formality == 17


def test_plain_messages_still_receive_distinct_neutral_contrasts() -> None:
    messages = parse_messages(
        "\n".join(
            [
                "You: The current plan has too many moving pieces.",
                "You: Summarize the decision and the next step.",
                "You: Show one concrete example.",
                "You: Test this with three users.",
            ]
        )
    )

    pairs = propose_exemplar_pairs(messages)

    assert all(pair.positive != pair.neutral for pair in pairs)


def test_persisted_capsule_contains_no_raw_or_unselected_messages() -> None:
    draft = build_ingestion_draft(SAMPLE, "Alex", True)
    approved = approve_draft(
        draft,
        {draft.proposed_pairs[0].pair_hash},
        {},
    )
    record = CapsuleRecord(
        capsule_id="capsule-1",
        owner_id="hf:alex",
        name="Alex Signal",
        status="profile_approved",
        style_profile=approved.profile,
        exemplar_pairs=approved.exemplar_pairs,
        source_fingerprint=approved.source_fingerprint,
    )
    repository = InMemoryCapsuleRepository([record])

    persisted = repository.get_for_owner("hf:alex", "capsule-1")

    assert len(persisted.exemplar_pairs) == 1
    assert not hasattr(persisted, "raw_input")
    assert SAMPLE not in repr(persisted)
