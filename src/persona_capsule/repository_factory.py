"""Select durable capsule storage without coupling callers to a provider."""

from persona_capsule.config import Settings
from persona_capsule.repository import (
    CapsuleRepository,
    FileCapsuleRepository,
    HuggingFaceDatasetCapsuleRepository,
)


def build_capsule_repository(settings: Settings) -> CapsuleRepository:
    if settings.hf_capsule_repo_id:
        if not settings.hugging_face_available:
            raise RuntimeError("HF_CAPSULE_REPO_ID requires HF_TOKEN.")
        return HuggingFaceDatasetCapsuleRepository(
            settings.hf_capsule_repo_id,
            token_from_settings(settings),
        )
    return FileCapsuleRepository(settings.capsule_data_dir)


def token_from_settings(settings: Settings) -> str:
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for Hugging Face capsule storage.")
    return token
