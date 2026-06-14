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
