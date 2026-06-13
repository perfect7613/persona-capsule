"""Safe PersonaSpec-style capsule export and compatibility manifest."""

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from persona_capsule.repository import CapsuleRecord
from persona_capsule.steering import SteeringRecipe

PERSONA_EXPORT_VERSION = "0.1"


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ).encode()


@dataclass(frozen=True, slots=True)
class CapsuleExportBundle:
    persona_filename: str
    persona_bytes: bytes
    manifest_filename: str
    manifest_bytes: bytes


def build_capsule_export(
    record: CapsuleRecord,
    *,
    include_private_exemplars: bool = False,
) -> CapsuleExportBundle:
    if record.style_profile is None:
        raise ValueError("A completed profile is required for export.")
    recipe = SteeringRecipe()
    persona: dict[str, Any] = {
        "format": "persona",
        "format_version": PERSONA_EXPORT_VERSION,
        "capsule_id": record.capsule_id,
        "name": record.name,
        "communication_style": record.style_profile.as_dict(include_private_evidence=False),
        "permissions": {
            "private_by_default": True,
            "contains_private_exemplars": include_private_exemplars,
            "activation_tensor_included": False,
        },
        "attribution": {
            "generator": "Persona Capsule",
            "source_fingerprint": record.source_fingerprint,
        },
    }
    if include_private_exemplars:
        persona["private_exemplar_pairs"] = [
            {
                "positive": pair.positive,
                "neutral": pair.neutral,
                "source_index": pair.source_index,
            }
            for pair in record.exemplar_pairs
        ]

    persona_bytes = _canonical_json(persona)
    manifest = {
        "format": "persona-capsule-compatibility-manifest",
        "format_version": "1",
        "capsule_id": record.capsule_id,
        "persona_sha256": sha256(persona_bytes).hexdigest(),
        "steering_recipe": recipe.as_dict(),
        "visual_recipe": {
            "provider": record.card_provider,
            "model_id": record.card_model_id,
            "model_revision": record.card_model_revision,
            "prompt_hash": record.card_prompt_hash,
            "seed": record.card_seed,
        },
        "exemplar_integrity": {
            "count": len(record.exemplar_pairs),
            "pair_hashes": [pair.pair_hash for pair in record.exemplar_pairs],
        },
        "artifacts": [
            {
                "artifact_id": sha256(reference.encode()).hexdigest(),
                "kind": reference.split("/", 1)[0] if "/" in reference else "managed",
                "integrity": "managed-by-capsule-repository",
            }
            for reference in record.artifact_refs
        ],
        "security": {
            "serialized_activation_tensor": False,
            "executable_pickle": False,
            "internal_paths_included": False,
        },
    }
    manifest_bytes = _canonical_json(manifest)
    slug = "".join(character if character.isalnum() else "-" for character in record.name)
    slug = "-".join(part for part in slug.split("-") if part).lower() or "capsule"
    return CapsuleExportBundle(
        persona_filename=f"{slug}.persona",
        persona_bytes=persona_bytes,
        manifest_filename=f"{slug}.manifest.json",
        manifest_bytes=manifest_bytes,
    )
