import json

from persona_capsule.export import build_capsule_export
from persona_capsule.profile import ExemplarPair, StyleDimensions, StyleProfile
from persona_capsule.repository import CapsuleRecord
from persona_capsule.steering import MODEL_ID, MODEL_REVISION


def _record() -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id="capsule-export",
        owner_id="hf:owner",
        name="Clear Signal",
        status="profile_approved",
        style_profile=StyleProfile(
            summary="Warm and direct.",
            descriptors=("warm", "direct"),
            lexical_tendencies=("concrete verbs",),
            sentence_rhythm="Concise.",
            dimensions=StyleDimensions(50, 70, 60, 65, 40, 85, 30),
            evidence=("Private evidence.",),
            uncertainty=0.2,
        ),
        exemplar_pairs=(ExemplarPair("Good. Test it now.", "Test it now.", 0),),
        source_fingerprint="source-v1",
        artifact_refs=("/Users/owner/private/card.png",),
        voice_provider="openbmb-voxcpm2-modal",
        voice_id="private-provider-voice-id",
        voice_status="ready",
        voice_retention="retained",
        voice_model_id="eleven_multilingual_v2",
    )


def test_default_export_excludes_private_data_and_activation_tensors() -> None:
    bundle = build_capsule_export(_record())
    persona = json.loads(bundle.persona_bytes)
    manifest = json.loads(bundle.manifest_bytes)
    serialized = bundle.persona_bytes + bundle.manifest_bytes

    assert bundle.persona_filename.endswith(".persona")
    assert "private_exemplar_pairs" not in persona
    assert "Private evidence." not in serialized.decode()
    assert persona["permissions"]["activation_tensor_included"] is False
    assert manifest["steering_recipe"]["model_id"] == MODEL_ID
    assert manifest["steering_recipe"]["model_revision"] == MODEL_REVISION
    assert manifest["security"]["serialized_activation_tensor"] is False
    assert manifest["visual_recipe"]["model_id"] == ""
    assert manifest["voice_recipe"]["provider"] == "openbmb-voxcpm2-modal"
    assert manifest["voice_recipe"]["provider_voice_id_included"] is False
    assert "private-provider-voice-id" not in serialized.decode()
    assert manifest["persona_sha256"]
    assert "/Users/" not in serialized.decode()
    assert "private/card.png" not in serialized.decode()
    assert "HF_TOKEN" not in serialized.decode()


def test_private_exemplars_require_explicit_export_selection() -> None:
    bundle = build_capsule_export(_record(), include_private_exemplars=True)
    persona = json.loads(bundle.persona_bytes)

    assert persona["permissions"]["contains_private_exemplars"] is True
    assert persona["private_exemplar_pairs"][0]["positive"] == "Good. Test it now."
