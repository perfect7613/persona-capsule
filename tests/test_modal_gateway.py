from persona_capsule.modal_gateway import ModalSteeringGateway
from persona_capsule.profile import ExemplarPair


def test_gateway_repairs_identical_legacy_pairs_before_remote_call() -> None:
    payload = ModalSteeringGateway._pairs_payload(
        (
            ExemplarPair(
                positive="The current plan has too many moving pieces.",
                neutral="The current plan has too many moving pieces.",
                source_index=3,
            ),
        )
    )

    assert payload[0]["positive"] != payload[0]["neutral"]
    assert payload[0]["legacy_contrast_repaired"] is True


def test_gateway_preserves_existing_distinct_pairs() -> None:
    payload = ModalSteeringGateway._pairs_payload(
        (
            ExemplarPair(
                positive="Good progress! Test it now.",
                neutral="Test it now.",
                source_index=1,
            ),
        )
    )

    assert payload[0]["neutral"] == "Test it now."
    assert "legacy_contrast_repaired" not in payload[0]


def test_gateway_evenly_bounds_legacy_fusion_pairs_to_runtime_limit() -> None:
    pairs = tuple(
        ExemplarPair(
            positive=f"Styled example {index}.",
            neutral=f"Neutral example {index}.",
            source_index=index,
        )
        for index in range(8)
    )

    payload = ModalSteeringGateway._pairs_payload(pairs)

    assert len(payload) == 6
    assert payload[0]["source_index"] == 0
    assert payload[-1]["source_index"] == 7
    assert any(item["source_index"] < 4 for item in payload)
    assert any(item["source_index"] >= 4 for item in payload)
