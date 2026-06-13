"""Environment-backed application configuration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Secret-safe settings derived from the process environment."""

    app_env: str = "development"
    hugging_face_available: bool = False
    modal_available: bool = False
    elevenlabs_available: bool = False
    space_environment: bool = False
    local_oauth_available: bool = False
    local_identity_enabled: bool = False
    local_hf_username: str = ""
    capsule_data_dir: str = ".persona-capsule-data"
    hf_capsule_repo_id: str = ""
    public_base_url: str = "http://127.0.0.1:7860"
    voice_temporary_hours: int = 24

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        if environ is None:
            load_dotenv()
            environ = os.environ

        return cls(
            app_env=environ.get("APP_ENV", "development").strip() or "development",
            hugging_face_available=bool(environ.get("HF_TOKEN", "").strip()),
            modal_available=all(
                environ.get(name, "").strip() for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
            ),
            elevenlabs_available=bool(environ.get("ELEVENLABS_API_KEY", "").strip()),
            space_environment=bool(environ.get("SPACE_ID", "").strip()),
            local_oauth_available=bool(environ.get("HF_TOKEN", "").strip()),
            local_identity_enabled=environ.get("PERSONA_LOCAL_IDENTITY", "").lower()
            in {"1", "true", "yes"},
            local_hf_username=environ.get("PERSONA_LOCAL_HF_USERNAME", "").strip(),
            capsule_data_dir=environ.get(
                "PERSONA_CAPSULE_DATA_DIR",
                ".persona-capsule-data",
            ).strip()
            or ".persona-capsule-data",
            hf_capsule_repo_id=environ.get("HF_CAPSULE_REPO_ID", "").strip(),
            public_base_url=environ.get(
                "PUBLIC_BASE_URL",
                "http://127.0.0.1:7860",
            ).rstrip("/"),
            voice_temporary_hours=max(
                1,
                int(environ.get("VOICE_TEMPORARY_HOURS", "24")),
            ),
        )

    @property
    def providers(self) -> dict[str, bool]:
        return {
            "hugging_face": self.hugging_face_available,
            "modal": self.modal_available,
            "elevenlabs": self.elevenlabs_available,
        }

    @property
    def local_identity_allowed(self) -> bool:
        return (
            self.app_env in {"development", "test"}
            and self.local_identity_enabled
            and bool(self.local_hf_username)
        )

    @property
    def oauth_ui_available(self) -> bool:
        return self.space_environment or self.local_oauth_available
