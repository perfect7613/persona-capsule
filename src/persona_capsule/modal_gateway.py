"""Thin client for the deployed Modal MiniCPM steering runtime."""

from typing import Any

import modal

from persona_capsule.profile import ExemplarPair
from persona_capsule.repository import CapsuleRecord

MODAL_APP_NAME = "persona-capsule-minicpm"
MODAL_CLASS_NAME = "MiniCPMSteeringRuntime"


class ModalSteeringGateway:
    """Resolve the deployed class lazily so app startup never loads the model."""

    def compare(
        self,
        *,
        owner_id: str,
        capsule_id: str,
        capsule_version: str,
        prompt: str,
        pairs: tuple[ExemplarPair, ...],
        strength: float,
    ) -> dict[str, Any]:
        runtime = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
        return runtime.compare.remote(
            owner_id=owner_id,
            capsule_id=capsule_id,
            capsule_version=capsule_version,
            prompt=prompt,
            pairs=[pair.as_dict() for pair in pairs],
            strength=strength,
            max_new_tokens=96,
        )

    def invalidate(self, *, owner_id: str, capsule_id: str) -> None:
        runtime = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
        runtime.invalidate.remote(owner_id=owner_id, capsule_id=capsule_id)

    def fuse(
        self,
        *,
        owner_id: str,
        first: CapsuleRecord,
        second: CapsuleRecord,
        prompt: str,
        first_weight: float,
        strength: float,
    ) -> dict[str, Any]:
        runtime = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
        return runtime.fuse.remote(
            owner_id=owner_id,
            first={
                "capsule_id": first.capsule_id,
                "capsule_version": first.source_fingerprint,
                "pairs": [pair.as_dict() for pair in first.exemplar_pairs],
            },
            second={
                "capsule_id": second.capsule_id,
                "capsule_version": second.source_fingerprint,
                "pairs": [pair.as_dict() for pair in second.exemplar_pairs],
            },
            prompt=prompt,
            first_weight=first_weight,
            strength=strength,
            max_new_tokens=120,
        )
