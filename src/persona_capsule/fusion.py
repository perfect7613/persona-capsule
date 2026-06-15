"""Owner-scoped capsule fusion with request-scoped vector composition."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from persona_capsule.card import CapsuleCardService
from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import (
    StyleDimensions,
    StyleProfile,
    bounded_exemplar_pairs,
)
from persona_capsule.repository import CapsuleRecord
from persona_capsule.steering import SteeringCompatibilityError, SteeringRecipe

VOICE_STRATEGIES = {"source_a", "source_b", "alternate", "none"}


class FusionGateway(Protocol):
    def fuse(
        self,
        *,
        owner_id: str,
        first: CapsuleRecord,
        second: CapsuleRecord,
        prompt: str,
        first_weight: float,
        strength: float,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FusionCompatibility:
    compatible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FusionResult:
    record: CapsuleRecord
    response: str
    diagnostics: dict[str, Any]
    interactive_card_path: str
    social_card_path: str


def _recipe(record: CapsuleRecord) -> dict[str, Any]:
    return record.steering_recipe or SteeringRecipe().as_dict()


def check_compatibility(first: CapsuleRecord, second: CapsuleRecord) -> FusionCompatibility:
    if first.capsule_id == second.capsule_id:
        return FusionCompatibility(False, "Choose two different capsules.")
    if not first.exemplar_pairs or not second.exemplar_pairs:
        return FusionCompatibility(False, "Both capsules need approved private steering pairs.")
    if _recipe(first) != _recipe(second):
        return FusionCompatibility(
            False,
            "Model revision, layer set, hidden size, aggregation, and normalization must match.",
        )
    return FusionCompatibility(True, "Exact MiniCPM steering recipes match.")


def merge_profiles(
    first: StyleProfile,
    second: StyleProfile,
    first_weight: float,
) -> StyleProfile:
    weight = float(first_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Fusion weight must be between 0 and 1.")
    second_weight = 1.0 - weight
    dimensions = {
        name: round(weight * value + second_weight * second.dimensions.as_dict()[name], 2)
        for name, value in first.dimensions.as_dict().items()
    }

    def weighted_union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
        ordered = (left, right) if weight >= 0.5 else (right, left)
        return tuple(dict.fromkeys((*ordered[0], *ordered[1])))

    first_percent = round(weight * 100)
    second_percent = 100 - first_percent
    return StyleProfile(
        summary=(
            f"A {first_percent}/{second_percent} communication-style fusion: "
            f"{first.summary.rstrip('.')} with {second.summary[:1].lower() + second.summary[1:]}"
        ),
        descriptors=weighted_union(first.descriptors, second.descriptors)[:8],
        lexical_tendencies=weighted_union(
            first.lexical_tendencies,
            second.lexical_tendencies,
        )[:8],
        sentence_rhythm=(
            f"{first.sentence_rhythm.rstrip('.')} blended with "
            f"{second.sentence_rhythm[:1].lower() + second.sentence_rhythm[1:]}"
        ),
        dimensions=StyleDimensions.from_dict(dimensions),
        evidence=(),
        uncertainty=round(
            weight * first.uncertainty + second_weight * second.uncertainty,
            4,
        ),
    )


class CapsuleFusionService:
    def __init__(
        self,
        capsule_library: CapsuleLibrary,
        gateway: FusionGateway,
        card_service: CapsuleCardService,
    ) -> None:
        self._capsule_library = capsule_library
        self._gateway = gateway
        self._card_service = card_service

    def compatibility(
        self,
        principal: Principal | None,
        first_id: str,
        second_id: str,
    ) -> FusionCompatibility:
        first = self._capsule_library.get_capsule(principal, first_id)
        second = self._capsule_library.get_capsule(principal, second_id)
        return check_compatibility(first, second)

    def create(
        self,
        principal: Principal | None,
        *,
        first_id: str,
        second_id: str,
        first_weight: float,
        prompt: str,
        name: str,
        voice_strategy: str,
        strength: float = 0.85,
    ) -> FusionResult:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        first = self._capsule_library.get_capsule(principal, first_id)
        second = self._capsule_library.get_capsule(principal, second_id)
        compatibility = check_compatibility(first, second)
        if not compatibility.compatible:
            raise SteeringCompatibilityError(compatibility.reason)
        if first.style_profile is None or second.style_profile is None:
            raise ValueError("Both sources need approved style profiles.")
        clean_name = name.strip()
        clean_prompt = prompt.strip()
        if not clean_name:
            raise ValueError("Give the fused capsule a name.")
        if not clean_prompt:
            raise ValueError("Enter a prompt for the fused response.")
        if voice_strategy not in VOICE_STRATEGIES:
            raise ValueError("Choose an explicit source voice strategy.")
        if voice_strategy == "source_a" and not first.voice_id:
            raise ValueError("The first capsule has no retained voice.")
        if voice_strategy == "source_b" and not second.voice_id:
            raise ValueError("The second capsule has no retained voice.")
        if voice_strategy == "alternate" and not (first.voice_id and second.voice_id):
            raise ValueError("Alternating voice requires retained voices on both sources.")

        weight = float(first_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Fusion weight must be between 0 and 1.")
        generated = self._gateway.fuse(
            owner_id=principal.user_id,
            first=first,
            second=second,
            prompt=clean_prompt,
            first_weight=weight,
            strength=float(strength),
        )
        profile = merge_profiles(first.style_profile, second.style_profile, weight)
        provenance = {
            "kind": "fusion",
            "format_version": "persona-fusion-v1",
            "sources": [
                {
                    "capsule_id": first.capsule_id,
                    "version": first.source_fingerprint,
                    "weight": weight,
                    "name": first.name,
                    "recipe": _recipe(first),
                    "exemplar_pairs": [pair.as_dict() for pair in first.exemplar_pairs],
                },
                {
                    "capsule_id": second.capsule_id,
                    "version": second.source_fingerprint,
                    "weight": 1.0 - weight,
                    "name": second.name,
                    "recipe": _recipe(second),
                    "exemplar_pairs": [pair.as_dict() for pair in second.exemplar_pairs],
                },
            ],
            "voice_strategy": voice_strategy,
            "persisted_tensor": False,
        }
        fingerprint = sha256(
            (
                f"{first.capsule_id}:{first.source_fingerprint}:{weight}:"
                f"{second.capsule_id}:{second.source_fingerprint}:{1.0 - weight}"
            ).encode()
        ).hexdigest()
        voice_source = {
            "source_a": first,
            "source_b": second,
        }.get(voice_strategy)
        record = self._capsule_library.save_capsule(
            principal,
            CapsuleRecord(
                capsule_id=uuid4().hex,
                owner_id=principal.user_id,
                name=clean_name,
                status="profile_approved",
                style_profile=profile,
                exemplar_pairs=bounded_exemplar_pairs(
                    (*first.exemplar_pairs, *second.exemplar_pairs)
                ),
                source_fingerprint=fingerprint,
                steering_recipe=_recipe(first),
                provenance=provenance,
                voice_provider=voice_source.voice_provider if voice_source else "",
                voice_id=voice_source.voice_id if voice_source else "",
                voice_status=voice_source.voice_status if voice_source else "",
                voice_retention=voice_source.voice_retention if voice_source else "",
                voice_consent_at=voice_source.voice_consent_at if voice_source else "",
                voice_expires_at=voice_source.voice_expires_at if voice_source else "",
                voice_sample_ref=voice_source.voice_sample_ref if voice_source else "",
                voice_model_id=voice_source.voice_model_id if voice_source else "",
            ),
        )
        card = self._card_service.generate(
            principal,
            record.capsule_id,
            variation="kinetic",
        )
        return FusionResult(
            record=card.record,
            response=str(generated["fused"]),
            diagnostics=dict(generated["diagnostics"]),
            interactive_card_path=str(card.interactive_path),
            social_card_path=str(card.social_path),
        )

    def source_availability(
        self,
        principal: Principal | None,
        fusion_id: str,
    ) -> tuple[dict[str, Any], ...]:
        fusion = self._capsule_library.get_capsule(principal, fusion_id)
        provenance = fusion.provenance or {}
        sources = provenance.get("sources", ())
        available_ids = {
            record.capsule_id for record in self._capsule_library.list_capsules(principal)
        }
        return tuple(
            {**dict(source), "available": source.get("capsule_id") in available_ids}
            for source in sources
        )
