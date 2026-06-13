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
    enable_creation: bool = True
    enable_steering: bool = True
    enable_card: bool = True
    enable_voice: bool = True
    enable_fusion: bool = True
    enable_battle: bool = True
    enable_deep_training: bool = True
    quota_steering_daily: int = 20
    quota_creation_daily: int = 10
    quota_card_daily: int = 10
    quota_voice_daily: int = 5
    quota_fusion_daily: int = 8
    quota_battle_daily: int = 6
    quota_deep_training_daily: int = 2

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        if environ is None:
            load_dotenv()
            environ = os.environ

        def enabled(name: str, default: str = "true") -> bool:
            return environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

        def quota(name: str, default: int) -> int:
            return max(0, int(environ.get(name, str(default))))

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
            enable_creation=enabled("ENABLE_CREATION"),
            enable_steering=enabled("ENABLE_STEERING"),
            enable_card=enabled("ENABLE_CARD"),
            enable_voice=enabled("ENABLE_VOICE"),
            enable_fusion=enabled("ENABLE_FUSION"),
            enable_battle=enabled("ENABLE_BATTLE"),
            enable_deep_training=enabled("ENABLE_DEEP_TRAINING"),
            quota_steering_daily=quota("QUOTA_STEERING_DAILY", 20),
            quota_creation_daily=quota("QUOTA_CREATION_DAILY", 10),
            quota_card_daily=quota("QUOTA_CARD_DAILY", 10),
            quota_voice_daily=quota("QUOTA_VOICE_DAILY", 5),
            quota_fusion_daily=quota("QUOTA_FUSION_DAILY", 8),
            quota_battle_daily=quota("QUOTA_BATTLE_DAILY", 6),
            quota_deep_training_daily=quota("QUOTA_DEEP_TRAINING_DAILY", 2),
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

    @property
    def feature_flags(self) -> dict[str, bool]:
        return {
            "creation": self.enable_creation,
            "steering": self.enable_steering,
            "card": self.enable_card,
            "voice": self.enable_voice,
            "fusion": self.enable_fusion,
            "battle": self.enable_battle,
            "deep_training": self.enable_deep_training,
        }

    @property
    def quotas(self) -> dict[str, int]:
        return {
            "creation": self.quota_creation_daily,
            "steering": self.quota_steering_daily,
            "card": self.quota_card_daily,
            "voice": self.quota_voice_daily,
            "fusion": self.quota_fusion_daily,
            "battle": self.quota_battle_daily,
            "deep_training": self.quota_deep_training_daily,
        }
