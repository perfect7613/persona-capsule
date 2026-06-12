"""Hugging Face and local-development identity resolution."""

from dataclasses import dataclass
from typing import Protocol

from persona_capsule.config import Settings


class OAuthProfileLike(Protocol):
    username: str


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    username: str
    source: str


class IdentityGateway:
    """Resolve trusted identities without creating anonymous principals."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def resolve(self, profile: OAuthProfileLike | None) -> Principal | None:
        if profile is not None and profile.username.strip():
            username = profile.username.strip()
            return Principal(
                user_id=f"hf:{username.casefold()}",
                username=username,
                source="hugging_face_oauth",
            )
        return self.resolve_local()

    def resolve_local(self) -> Principal | None:
        if not self._settings.local_identity_allowed:
            return None

        username = self._settings.local_hf_username
        return Principal(
            user_id=f"hf:{username.casefold()}",
            username=username,
            source="local_development",
        )
