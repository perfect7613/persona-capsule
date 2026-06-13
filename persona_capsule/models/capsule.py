"""Capsule record schema."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from persona_capsule.models.profile import StyleProfile


class CreationMode(str, Enum):
    QUICK = "quick"
    DEEP = "deep"
    FUSION = "fusion"
    DEMO = "demo"


class CapsuleState(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    PUBLISHED = "published"
    DELETED = "deleted"


class Visibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ExemplarPair:
    style_example: str
    neutral_contrast: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> ExemplarPair:
        return cls(**data)


@dataclass
class SteeringRecipe:
    model_id: str = "openbmb/MiniCPM4.1-8B"
    model_revision: str = "main"
    layer_indices: list[int] = field(default_factory=lambda: [10, 18, 26])
    aggregation: str = "mean"
    default_strength: float = 0.65
    normalization: str = "per_layer_l2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SteeringRecipe:
        return cls(**data)


@dataclass
class CapsuleRecord:
    id: str
    owner_id: str
    display_name: str
    creation_mode: CreationMode
    state: CapsuleState
    visibility: Visibility
    profile: StyleProfile
    exemplars: list[ExemplarPair]
    steering_recipe: SteeringRecipe
    card_image_path: str | None = None
    share_image_path: str | None = None
    voice_sample_path: str | None = None
    public_slug: str | None = None
    provenance: list[str] = field(default_factory=list)
    fusion_weights: dict[str, float] = field(default_factory=dict)
    visual_lora_job_id: str | None = None
    visual_lora_path: str | None = None
    visual_trigger_word: str | None = None
    visual_training_status: str | None = None
    dataset_folder_path: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def new(
        cls,
        owner_id: str,
        display_name: str,
        profile: StyleProfile,
        exemplars: list[ExemplarPair],
        creation_mode: CreationMode = CreationMode.QUICK,
    ) -> CapsuleRecord:
        return cls(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            display_name=display_name,
            creation_mode=creation_mode,
            state=CapsuleState.READY,
            visibility=Visibility.PRIVATE,
            profile=profile,
            exemplars=exemplars,
            steering_recipe=SteeringRecipe(),
        )

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "display_name": self.display_name,
            "creation_mode": self.creation_mode.value,
            "state": self.state.value,
            "visibility": self.visibility.value,
            "profile": self.profile.to_dict(),
            "exemplars": [e.to_dict() for e in self.exemplars],
            "steering_recipe": self.steering_recipe.to_dict(),
            "card_image_path": self.card_image_path,
            "share_image_path": self.share_image_path,
            "voice_sample_path": self.voice_sample_path,
            "public_slug": self.public_slug,
            "provenance": self.provenance,
            "fusion_weights": self.fusion_weights,
            "visual_lora_job_id": self.visual_lora_job_id,
            "visual_lora_path": self.visual_lora_path,
            "visual_trigger_word": self.visual_trigger_word,
            "visual_training_status": self.visual_training_status,
            "dataset_folder_path": self.dataset_folder_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapsuleRecord:
        return cls(
            id=data["id"],
            owner_id=data["owner_id"],
            display_name=data["display_name"],
            creation_mode=CreationMode(data["creation_mode"]),
            state=CapsuleState(data["state"]),
            visibility=Visibility(data["visibility"]),
            profile=StyleProfile.from_dict(data["profile"]),
            exemplars=[ExemplarPair.from_dict(e) for e in data.get("exemplars", [])],
            steering_recipe=SteeringRecipe.from_dict(data.get("steering_recipe", {})),
            card_image_path=data.get("card_image_path"),
            share_image_path=data.get("share_image_path"),
            voice_sample_path=data.get("voice_sample_path"),
            public_slug=data.get("public_slug"),
            provenance=list(data.get("provenance", [])),
            fusion_weights=dict(data.get("fusion_weights", {})),
            visual_lora_job_id=data.get("visual_lora_job_id"),
            visual_lora_path=data.get("visual_lora_path"),
            visual_trigger_word=data.get("visual_trigger_word"),
            visual_training_status=data.get("visual_training_status"),
            dataset_folder_path=data.get("dataset_folder_path"),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
        )

    def public_projection(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "summary": self.profile.summary,
            "tone": self.profile.tone,
            "palette": self.profile.palette,
            "public_slug": self.public_slug,
            "card_image_path": self.share_image_path or self.card_image_path,
            "signature_phrases": self.profile.signature_phrases[:3],
            "creation_mode": self.creation_mode.value,
        }
