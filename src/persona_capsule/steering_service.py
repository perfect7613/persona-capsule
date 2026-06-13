"""Authenticated application service for live capsule steering."""

from typing import Any, Protocol

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import ExemplarPair
from persona_capsule.repository import CapsuleRecord
from persona_capsule.steering import validate_strength


class SteeringGateway(Protocol):
    def compare(
        self,
        *,
        owner_id: str,
        capsule_id: str,
        capsule_version: str,
        prompt: str,
        pairs: tuple[ExemplarPair, ...],
        strength: float,
    ) -> dict[str, Any]: ...

    def invalidate(self, *, owner_id: str, capsule_id: str) -> None: ...


class CapsuleSteeringService:
    def __init__(
        self,
        capsule_library: CapsuleLibrary,
        gateway: SteeringGateway,
    ) -> None:
        self._capsule_library = capsule_library
        self._gateway = gateway

    def compare(
        self,
        principal: Principal | None,
        capsule: CapsuleRecord | None,
        prompt: str,
        strength: float,
    ) -> dict[str, Any]:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        if capsule is None:
            raise ValueError("Approve a capsule before running live steering.")

        persisted = self._capsule_library.get_capsule(principal, capsule.capsule_id)
        if persisted.status != "profile_approved" or not persisted.exemplar_pairs:
            raise ValueError("This capsule has no approved private steering pairs.")
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Enter a prompt to compare.")

        return self._gateway.compare(
            owner_id=principal.user_id,
            capsule_id=persisted.capsule_id,
            capsule_version=persisted.source_fingerprint,
            prompt=cleaned_prompt,
            pairs=persisted.exemplar_pairs,
            strength=validate_strength(strength),
        )

    def invalidate(self, principal: Principal | None, capsule_id: str) -> None:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        self._gateway.invalidate(
            owner_id=principal.user_id,
            capsule_id=capsule_id,
        )
