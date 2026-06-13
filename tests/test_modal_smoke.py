import os

import pytest

from persona_capsule.deep_gateway import ModalDeepTrainingGateway
from persona_capsule.modal_gateway import ModalSteeringGateway
from persona_capsule.nemotron_gateway import ModalNemotronGateway
from persona_capsule.profile import ExemplarPair


@pytest.mark.skipif(
    os.environ.get("RUN_MODAL_SMOKE") != "1",
    reason="Set RUN_MODAL_SMOKE=1 after deploying Modal runtimes with rotated credentials.",
)
def test_live_minicpm_and_nemotron_smoke() -> None:
    pair = ExemplarPair(
        positive="Clear point. One reason, one example, then the next action.",
        neutral="Provide a response.",
        source_index=0,
    )
    comparison = ModalSteeringGateway().compare(
        owner_id="smoke-test",
        capsule_id="smoke-test",
        capsule_version="v1",
        prompt="Explain why a small test is useful.",
        pairs=(pair,),
        strength=0.5,
    )
    assert comparison["baseline"]
    assert comparison["steered"]

    judgment = ModalNemotronGateway(attempts=1).judge(
        {
            "challenge": "Explain why a small test is useful.",
            "candidate_a": comparison["baseline"],
            "candidate_b": comparison["steered"],
            "rubric": ["style_fidelity", "response_quality", "instruction_adherence", "safety"],
        }
    )
    assert judgment["scores"]["candidate_a"]
    assert judgment["scores"]["candidate_b"]


@pytest.mark.skipif(
    os.environ.get("RUN_DEEP_MODAL_SMOKE") != "1",
    reason="Set RUN_DEEP_MODAL_SMOKE=1 only when a billed start-and-cancel smoke is intended.",
)
def test_live_deep_job_can_start_and_cancel() -> None:
    gateway = ModalDeepTrainingGateway()
    call_id = gateway.spawn(
        {
            "schema_version": "persona-deep-v1",
            "owner_namespace": "smoke-test",
            "capsule_id": "smoke-test",
            "capsule_version": "v1",
            "profile": {},
            "training_pairs": [
                {"positive": "Clear and warm.", "neutral": "Clear.", "source_index": 0},
                {"positive": "Test it now.", "neutral": "Test it.", "source_index": 1},
            ],
            "visual_lora": False,
            "idempotency_key": "smoke-test-cancel",
        }
    )
    assert call_id
    gateway.cancel(call_id)
