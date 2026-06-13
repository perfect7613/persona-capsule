"""Authenticated private capsule-library operations."""

from persona_capsule.identity import Principal
from persona_capsule.repository import CapsuleRecord, CapsuleRepository


class AuthenticationRequiredError(PermissionError):
    """Raised when a creator operation has no resolved identity."""


class CapsuleLibrary:
    def __init__(self, repository: CapsuleRepository) -> None:
        self._repository = repository

    @staticmethod
    def _owner(principal: Principal | None) -> str:
        if principal is None:
            raise AuthenticationRequiredError("Hugging Face login required")
        return principal.user_id

    def list_capsules(self, principal: Principal | None) -> tuple[CapsuleRecord, ...]:
        return self._repository.list_for_owner(self._owner(principal))

    def get_capsule(self, principal: Principal | None, capsule_id: str) -> CapsuleRecord:
        return self._repository.get_for_owner(self._owner(principal), capsule_id)

    def save_capsule(
        self,
        principal: Principal | None,
        record: CapsuleRecord,
    ) -> CapsuleRecord:
        return self._repository.save_for_owner(self._owner(principal), record)

    def rename_capsule(
        self,
        principal: Principal | None,
        capsule_id: str,
        name: str,
    ) -> CapsuleRecord:
        return self._repository.rename_for_owner(self._owner(principal), capsule_id, name)

    def delete_capsule(
        self,
        principal: Principal | None,
        capsule_id: str,
    ) -> bool:
        return self._repository.delete_for_owner(self._owner(principal), capsule_id)
