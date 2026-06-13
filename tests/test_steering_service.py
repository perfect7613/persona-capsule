import pytest

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import ExemplarPair
from persona_capsule.repository import (
    CapsuleNotFoundError,
    CapsuleRecord,
    InMemoryCapsuleRepository,
)
from persona_capsule.steering_service import CapsuleSteeringService

ALICE = Principal("hf:alice", "alice", "test")
BOB = Principal("hf:bob", "bob", "test")
PAIR = ExemplarPair("Warm and direct.", "The response is direct.", 0)


class RecordingGateway:
    def __init__(self) -> None:
        self.calls = []
        self.invalidations = []

    def compare(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "baseline": "baseline",
            "steered": "steered",
            "diagnostics": {"layers": []},
        }

    def invalidate(self, **kwargs):
        self.invalidations.append(kwargs)


def _service() -> tuple[CapsuleSteeringService, RecordingGateway, CapsuleRecord]:
    record = CapsuleRecord(
        capsule_id="capsule-1",
        owner_id=ALICE.user_id,
        name="Alice Signal",
        status="profile_approved",
        exemplar_pairs=(PAIR,),
        source_fingerprint="version-1",
    )
    gateway = RecordingGateway()
    service = CapsuleSteeringService(
        CapsuleLibrary(InMemoryCapsuleRepository([record])),
        gateway,
    )
    return service, gateway, record


def test_compare_sends_only_approved_owner_scoped_material() -> None:
    service, gateway, record = _service()

    result = service.compare(ALICE, record, "  Explain the decision.  ", 0.85)

    assert result["steered"] == "steered"
    assert gateway.calls == [
        {
            "owner_id": ALICE.user_id,
            "capsule_id": record.capsule_id,
            "capsule_version": "version-1",
            "prompt": "Explain the decision.",
            "pairs": (PAIR,),
            "strength": 0.85,
        }
    ]


def test_compare_rejects_anonymous_cross_owner_and_unapproved_requests() -> None:
    service, gateway, record = _service()

    with pytest.raises(PermissionError):
        service.compare(None, record, "Prompt", 0.8)
    with pytest.raises(CapsuleNotFoundError):
        service.compare(BOB, record, "Prompt", 0.8)
    with pytest.raises(ValueError, match="Approve a capsule"):
        service.compare(ALICE, None, "Prompt", 0.8)
    with pytest.raises(ValueError, match="Enter a prompt"):
        service.compare(ALICE, record, " ", 0.8)

    assert gateway.calls == []


def test_invalidate_is_owner_scoped() -> None:
    service, gateway, record = _service()

    service.invalidate(ALICE, record.capsule_id)

    assert gateway.invalidations == [{"owner_id": ALICE.user_id, "capsule_id": record.capsule_id}]
