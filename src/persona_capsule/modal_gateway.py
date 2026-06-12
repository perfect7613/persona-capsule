"""Thin client for the deployed Modal MiniCPM steering runtime."""

from typing import Any

import modal

from persona_capsule.profile import ExemplarPair

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
