import pytest

from persona_capsule.deep_training import DeepCapsuleService, estimate_deep_training
from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import ExemplarPair
from persona_capsule.repository import CapsuleRecord, InMemoryCapsuleRepository

PRINCIPAL = Principal("hf:alice", "alice", "test")


def _record() -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id="abc123",
        owner_id=PRINCIPAL.user_id,
        name="Quick Capsule",
        status="profile_approved",
        exemplar_pairs=(
            ExemplarPair("Warm and direct.", "Direct.", 0),
            ExemplarPair("Good. Test it now.", "Test it now.", 1),
        ),
        source_fingerprint="quick-v1",
    )


class FakeDeepGateway:
    def __init__(self) -> None:
        self.spawned = []
        self.cancelled = []
        self.result = {"status": "running", "progress": 50, "message": "Training."}

    def spawn(self, payload):
        self.spawned.append(payload)
        return "fc-123"

    def poll(self, call_id):
        assert call_id == "fc-123"
        return self.result

    def cancel(self, call_id):
        self.cancelled.append(call_id)


def test_estimate_and_idempotent_start_preserve_quick_capsule() -> None:
    library = CapsuleLibrary(InMemoryCapsuleRepository([_record()]))
    gateway = FakeDeepGateway()
    service = DeepCapsuleService(library, gateway)

    estimate = estimate_deep_training(visual_lora=True)
    first = service.start(
        PRINCIPAL,
        "abc123",
        idempotency_key="deep-request-001",
        visual_lora=True,
        confirmed=True,
    )
    second = service.start(
        PRINCIPAL,
        "abc123",
        idempotency_key="deep-request-001",
        visual_lora=True,
        confirmed=True,
    )

    assert estimate.estimated_minutes == 35
    assert estimate.estimated_modal_credits == 8.0
    assert first == second
    assert len(gateway.spawned) == 1
    assert first.status == "profile_approved"
    assert first.deep_training["status"] == "queued"


def test_poll_attaches_only_a_passing_safetensors_artifact() -> None:
    library = CapsuleLibrary(InMemoryCapsuleRepository([_record()]))
    gateway = FakeDeepGateway()
    service = DeepCapsuleService(library, gateway)
    service.start(
        PRINCIPAL,
        "abc123",
        idempotency_key="deep-request-001",
        visual_lora=False,
        confirmed=True,
    )
    gateway.result = {
        "status": "completed",
        "progress": 100,
        "evaluation": {"quality_gain": 0.11, "memorization_score": 0.1},
        "artifact": {
            "repo_id": "alice/private-lora",
            "revision": "sha123",
            "format": "peft-safetensors",
        },
    }

    completed = service.poll(PRINCIPAL, "abc123")

    assert completed.deep_training["artifact_attached"] is True
    assert completed.deep_training["artifact"]["format"] == "peft-safetensors"
    assert completed.status == "profile_approved"


def test_failed_evaluation_and_cancellation_leave_quick_capsule_usable() -> None:
    library = CapsuleLibrary(InMemoryCapsuleRepository([_record()]))
    gateway = FakeDeepGateway()
    service = DeepCapsuleService(library, gateway)
    service.start(
        PRINCIPAL,
        "abc123",
        idempotency_key="deep-request-001",
        visual_lora=False,
        confirmed=True,
    )
    gateway.result = {
        "status": "completed",
        "progress": 100,
        "evaluation": {"quality_gain": 0.01, "memorization_score": 0.4},
        "artifact": {"repo_id": "should/not-attach", "revision": "sha"},
    }

    evaluated = service.poll(PRINCIPAL, "abc123")

    assert evaluated.deep_training["artifact_attached"] is False
    assert evaluated.status == "profile_approved"

    repository = InMemoryCapsuleRepository([_record()])
    cancel_service = DeepCapsuleService(CapsuleLibrary(repository), gateway)
    cancel_service.start(
        PRINCIPAL,
        "abc123",
        idempotency_key="deep-request-002",
        visual_lora=False,
        confirmed=True,
    )
    cancelled = cancel_service.cancel(PRINCIPAL, "abc123")
    assert gateway.cancelled == ["fc-123"]
    assert cancelled.deep_training["status"] == "cancelled"
    assert cancelled.status == "profile_approved"


def test_deep_training_requires_opt_in_and_identity() -> None:
    service = DeepCapsuleService(
        CapsuleLibrary(InMemoryCapsuleRepository([_record()])),
        FakeDeepGateway(),
    )
    with pytest.raises(ValueError, match="Confirm"):
        service.start(
            PRINCIPAL,
            "abc123",
            idempotency_key="deep-request-001",
            visual_lora=False,
            confirmed=False,
        )
    with pytest.raises(PermissionError):
        service.start(
            None,
            "abc123",
            idempotency_key="deep-request-001",
            visual_lora=False,
            confirmed=True,
        )


@pytest.mark.parametrize("status", ["failed", "timeout"])
def test_terminal_provider_states_are_resumable_and_attach_nothing(status: str) -> None:
    library = CapsuleLibrary(InMemoryCapsuleRepository([_record()]))
    gateway = FakeDeepGateway()
    service = DeepCapsuleService(library, gateway)
    service.start(
        PRINCIPAL,
        "abc123",
        idempotency_key="deep-request-terminal",
        visual_lora=False,
        confirmed=True,
    )
    gateway.result = {
        "status": status,
        "progress": 20,
        "message": f"Provider reported {status}.",
    }

    terminal = service.poll(PRINCIPAL, "abc123")
    resumed = service.poll(PRINCIPAL, "abc123")

    assert terminal.deep_training["status"] == status
    assert terminal.deep_training["artifact_attached"] is False
    assert resumed == terminal
    assert terminal.status == "profile_approved"
