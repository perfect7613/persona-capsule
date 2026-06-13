"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _mode() -> str:
    raw = os.getenv("PERSONA_CAPSULE_MODE", "auto").strip().lower()
    if raw in {"local", "production"}:
        return raw
    if os.getenv("HF_TOKEN") or os.getenv("MODAL_TOKEN_ID"):
        return "production"
    return "local"


@dataclass(frozen=True)
class Settings:
    mode: str
    data_dir: Path
    repo_root: Path
    host: str
    port: int
    dev_user_id: str
    dev_user_name: str
    hf_token: str | None
    modal_token_id: str | None
    modal_token_secret: str | None
    elevenlabs_api_key: str | None
    modal_dataset_path: str | None
    modal_model_volume: str
    flux_base_model: str
    default_training_steps: int

    @property
    def use_mock_providers(self) -> bool:
        return self.mode == "local"

    @property
    def capsules_dir(self) -> Path:
        return self.data_dir / "capsules"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def published_dir(self) -> Path:
        return self.data_dir / "published"

    @property
    def training_jobs_dir(self) -> Path:
        return self.data_dir / "training_jobs"

    @property
    def generated_configs_dir(self) -> Path:
        return self.repo_root / "config" / "generated"


def get_settings() -> Settings:
    data_dir = Path(os.getenv("PERSONA_CAPSULE_DATA_DIR", "./artifacts")).resolve()
    repo_root = Path(os.getenv("PERSONA_CAPSULE_REPO_ROOT", ".")).resolve()
    return Settings(
        mode=_mode(),
        data_dir=data_dir,
        repo_root=repo_root,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "7860")),
        dev_user_id=os.getenv("DEV_USER_ID", "dev-user"),
        dev_user_name=os.getenv("DEV_USER_NAME", "Local Developer"),
        hf_token=os.getenv("HF_TOKEN") or None,
        modal_token_id=os.getenv("MODAL_TOKEN_ID") or None,
        modal_token_secret=os.getenv("MODAL_TOKEN_SECRET") or None,
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
        modal_dataset_path=os.getenv("MODAL_DATASET_PATH") or None,
        modal_model_volume=os.getenv("MODAL_MODEL_VOLUME", "flux-lora-models"),
        flux_base_model=os.getenv(
            "FLUX_BASE_MODEL",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        default_training_steps=int(os.getenv("DEFAULT_TRAINING_STEPS", "1800")),
    )


def ensure_data_dirs(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.capsules_dir,
        settings.assets_dir,
        settings.published_dir,
        settings.training_jobs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
