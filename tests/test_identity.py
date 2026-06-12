from dataclasses import dataclass

from persona_capsule.config import Settings
from persona_capsule.identity import IdentityGateway


@dataclass
class Profile:
    username: str


def test_oauth_identity_takes_precedence_over_local_identity() -> None:
    gateway = IdentityGateway(
        Settings(
            app_env="development",
            local_identity_enabled=True,
            local_hf_username="local-owner",
        )
    )

    principal = gateway.resolve(Profile(username="SpaceOwner"))

    assert principal is not None
    assert principal.user_id == "hf:spaceowner"
    assert principal.username == "SpaceOwner"
    assert principal.source == "hugging_face_oauth"


def test_no_anonymous_principal_is_created() -> None:
    gateway = IdentityGateway(Settings(app_env="production"))

    assert gateway.resolve(None) is None
