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
        )

    @property
    def providers(self) -> dict[str, bool]:
        return {
            "hugging_face": self.hugging_face_available,
            "modal": self.modal_available,
            "elevenlabs": self.elevenlabs_available,
        }
