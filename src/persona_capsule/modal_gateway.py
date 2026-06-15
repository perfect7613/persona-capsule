"""Thin client for the deployed Modal MiniCPM steering runtime."""

from typing import Any

import modal

from persona_capsule.profile import ExemplarPair, ensure_distinct_contrast
from persona_capsule.repository import CapsuleRecord

MODAL_APP_NAME = "persona-capsule-minicpm"
MODAL_CLASS_NAME = "MiniCPMSteeringRuntimeV5"


class ModalSteeringGateway:
    """Resolve the deployed class lazily so app startup never loads the model."""

    @staticmethod
    def _pairs_payload(pairs: tuple[ExemplarPair, ...]) -> list[dict[str, Any]]:
        payload = []
        for pair in pairs:
            neutral, repaired = ensure_distinct_contrast(pair.positive, pair.neutral)
            item = {
                "positive": pair.positive,
                "neutral": neutral,
                "source_index": pair.source_index,
            }
            if repaired:
                item["legacy_contrast_repaired"] = True
            payload.append(item)
        return payload

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
            pairs=self._pairs_payload(pairs),
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
                "pairs": self._pairs_payload(first.exemplar_pairs),
            },
            second={
                "capsule_id": second.capsule_id,
                "capsule_version": second.source_fingerprint,
                "pairs": self._pairs_payload(second.exemplar_pairs),
            },
            prompt=prompt,
            first_weight=first_weight,
            strength=strength,
            max_new_tokens=120,
        )
