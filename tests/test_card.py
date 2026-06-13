import io
from hashlib import sha256
from pathlib import Path

from PIL import Image

from persona_capsule.card import (
    ART_SIZE,
    INTERACTIVE_SIZE,
    SOCIAL_SIZE,
    ArtResult,
    CapsuleCardService,
    DeterministicArtProvider,
    build_card_prompt,
    render_interactive_card,
    render_social_card,
)
from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import (
    ExemplarPair,
    StyleDimensions,
    StyleProfile,
)
from persona_capsule.repository import CapsuleRecord, InMemoryCapsuleRepository

OWNER = Principal("hf:owner", "owner", "test")
PRIVATE_SENTINEL = "PRIVATE-MESSAGE-SENTINEL"


def _record() -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id="card-01",
        owner_id=OWNER.user_id,
        name="Clear Signal",
        status="profile_approved",
        style_profile=StyleProfile(
            summary="Warm, direct, and analytical.",
            descriptors=("warm", "direct", "analytical", PRIVATE_SENTINEL),
            lexical_tendencies=("private lexical tendency",),
            sentence_rhythm="Short setup, clear landing.",
            dimensions=StyleDimensions(78, 84, 67, 71, 43, 91, 36),
            evidence=(PRIVATE_SENTINEL,),
            uncertainty=0.14,
        ),
        exemplar_pairs=(ExemplarPair(PRIVATE_SENTINEL, "A neutral sentence.", 0),),
        source_fingerprint="source-v1",
    )


def _library() -> tuple[CapsuleLibrary, CapsuleRecord]:
    repository = InMemoryCapsuleRepository()
    library = CapsuleLibrary(repository)
    record = library.save_capsule(OWNER, _record())
    return library, record


def test_prompt_uses_mapped_public_profile_fields_only() -> None:
    _, record = _library()

    prompt = build_card_prompt(record, variation="signal", seed=42)

    assert "glowing open aperture" in prompt.text
    assert "single precision beam" in prompt.text
    assert PRIVATE_SENTINEL not in prompt.text
    assert "private lexical tendency" not in prompt.text
    assert "no words" in prompt.text


def test_deterministic_fallback_and_card_dimensions() -> None:
    _, record = _library()
    prompt = build_card_prompt(record, variation="archive", seed=7613)
    provider = DeterministicArtProvider()

    first = provider.generate(prompt)
    second = provider.generate(prompt)
    art = Image.open(io.BytesIO(first.png_bytes))
    interactive = render_interactive_card(record, art)
    social = render_social_card(record, art)

    assert sha256(first.png_bytes).digest() == sha256(second.png_bytes).digest()
    assert art.size == ART_SIZE
    assert interactive.size == INTERACTIVE_SIZE
    assert social.size == SOCIAL_SIZE


class FailingProvider:
    def generate(self, prompt):
        del prompt
        raise RuntimeError("provider unavailable")


class SolidProvider:
    def generate(self, prompt):
        image = Image.new("RGB", ART_SIZE, prompt.palette[1])
        output = io.BytesIO()
        image.save(output, format="PNG")
        return ArtResult(
            png_bytes=output.getvalue(),
            provider="fake-flux",
            model_id="test/model",
            model_revision="revision",
        )


def test_service_falls_back_and_does_not_mutate_profile(tmp_path: Path) -> None:
    library, record = _library()
    service = CapsuleCardService(library, tmp_path, FailingProvider())

    result = service.generate(
        OWNER,
        record.capsule_id,
        variation="kinetic",
        seed=12,
    )
    reopened = library.get_capsule(OWNER, record.capsule_id)

    assert result.used_fallback is True
    assert result.provider == "deterministic-fallback"
    assert reopened.style_profile == record.style_profile
    assert reopened.card_image_ref.endswith("-interactive.png")
    assert reopened.social_image_ref.endswith("-social.png")
    assert Image.open(result.interactive_path).size == INTERACTIVE_SIZE
    assert Image.open(result.social_path).size == SOCIAL_SIZE


def test_controlled_regeneration_replaces_card_refs(tmp_path: Path) -> None:
    library, record = _library()
    service = CapsuleCardService(library, tmp_path, SolidProvider())

    first = service.generate(OWNER, record.capsule_id, variation="signal", seed=1)
    second = service.generate(OWNER, record.capsule_id, variation="archive", seed=2)

    assert first.used_fallback is False
    assert second.provider == "fake-flux"
    assert second.record.card_seed == 2
    assert second.record.card_prompt_hash != first.record.card_prompt_hash
    assert second.record.card_model_id == "test/model"
    assert second.record.card_model_revision == "revision"
    assert len(second.record.artifact_refs) == 2
    assert second.record.style_profile == record.style_profile
