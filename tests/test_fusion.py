from math import sqrt
from pathlib import Path

import pytest

from persona_capsule.card import CapsuleCardService
from persona_capsule.fusion import CapsuleFusionService, check_compatibility
from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import ExemplarPair, StyleDimensions, StyleProfile
from persona_capsule.repository import CapsuleRecord, InMemoryCapsuleRepository
from persona_capsule.steering import (
    SteeringCompatibilityError,
    SteeringRecipe,
    compose_layer_vectors,
)

PRINCIPAL = Principal("hf:alice", "alice", "test")


def _record(capsule_id: str, name: str, *, directness: float = 70) -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id=capsule_id,
        owner_id=PRINCIPAL.user_id,
        name=name,
        status="profile_approved",
        style_profile=StyleProfile(
            summary=f"{name} communicates with clarity.",
            descriptors=("direct", name.casefold()),
            lexical_tendencies=("concrete verbs",),
            sentence_rhythm="Short setup and a clean landing.",
            dimensions=StyleDimensions(70, 70, 60, 60, 45, directness, 35),
            evidence=("private",),
            uncertainty=0.2,
        ),
        exemplar_pairs=(
            ExemplarPair(f"{name}: test the smallest useful version.", "Test a small version.", 0),
        ),
        source_fingerprint=f"{capsule_id}-v1",
        steering_recipe=SteeringRecipe().as_dict(),
    )


class FakeFusionGateway:
    def __init__(self) -> None:
        self.calls = []

    def fuse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "fused": "A deterministic blended response.",
            "diagnostics": {
                "first_weight": kwargs["first_weight"],
                "persistence": "request-scoped vectors; no fused tensor written to storage",
            },
        }


def test_layer_composition_normalizes_sources_and_result() -> None:
    result = compose_layer_vectors({8: (10, 0)}, {8: (0, 2)}, 0.75)

    assert result[8] == pytest.approx((0.948683, 0.316228), abs=1e-5)
    assert sqrt(sum(value * value for value in result[8])) == pytest.approx(1)
    assert compose_layer_vectors({8: (10, 0)}, {8: (0, 2)}, 0.75) == result


def test_incompatible_recipe_is_rejected() -> None:
    first = _record("a1", "Alpha")
    second = CapsuleRecord.from_dict(
        {
            **_record("b2", "Beta").as_dict(),
            "steering_recipe": {**SteeringRecipe().as_dict(), "hidden_size": 123},
        }
    )

    compatibility = check_compatibility(first, second)

    assert compatibility.compatible is False
    assert "must match" in compatibility.reason


def test_fusion_saves_private_provenance_and_survives_source_deletion(tmp_path: Path) -> None:
    repository = InMemoryCapsuleRepository()
    library = CapsuleLibrary(repository)
    first = library.save_capsule(PRINCIPAL, _record("a1", "Alpha"))
    second = library.save_capsule(PRINCIPAL, _record("b2", "Beta", directness=40))
    gateway = FakeFusionGateway()
    service = CapsuleFusionService(
        library,
        gateway,
        CapsuleCardService(library, tmp_path),
    )

    result = service.create(
        PRINCIPAL,
        first_id=first.capsule_id,
        second_id=second.capsule_id,
        first_weight=0.6,
        prompt="Explain the next useful experiment.",
        name="Alpha Beta",
        voice_strategy="none",
    )

    assert result.response == "A deterministic blended response."
    assert result.record.provenance["persisted_tensor"] is False
    assert [source["weight"] for source in result.record.provenance["sources"]] == [0.6, 0.4]
    assert result.record.card_image_ref
    assert "vector" not in result.record.as_dict()
    assert gateway.calls[0]["first"].capsule_id == first.capsule_id

    library.delete_capsule(PRINCIPAL, first.capsule_id)
    availability = service.source_availability(PRINCIPAL, result.record.capsule_id)
    assert availability[0]["available"] is False
    assert availability[1]["available"] is True
    assert library.get_capsule(PRINCIPAL, result.record.capsule_id).name == "Alpha Beta"


def test_fusion_is_owner_scoped_and_requires_voice_availability(tmp_path: Path) -> None:
    repository = InMemoryCapsuleRepository([_record("a1", "Alpha"), _record("b2", "Beta")])
    service = CapsuleFusionService(
        CapsuleLibrary(repository),
        FakeFusionGateway(),
        CapsuleCardService(CapsuleLibrary(repository), tmp_path),
    )

    with pytest.raises(PermissionError):
        service.create(
            None,
            first_id="a1",
            second_id="b2",
            first_weight=0.5,
            prompt="A bounded prompt.",
            name="Fusion",
            voice_strategy="none",
        )
    with pytest.raises(ValueError, match="first capsule has no retained voice"):
        service.create(
            PRINCIPAL,
            first_id="a1",
            second_id="b2",
            first_weight=0.5,
            prompt="A bounded prompt.",
            name="Fusion",
            voice_strategy="source_a",
        )
    incompatible = CapsuleRecord.from_dict(
        {
            **_record("b2", "Beta").as_dict(),
            "steering_recipe": {**SteeringRecipe().as_dict(), "model_revision": "other"},
        }
    )
    repository.save_for_owner(PRINCIPAL.user_id, incompatible)
    with pytest.raises(SteeringCompatibilityError):
        service.create(
            PRINCIPAL,
            first_id="a1",
            second_id="b2",
            first_weight=0.5,
            prompt="A bounded prompt.",
            name="Fusion",
            voice_strategy="none",
        )
