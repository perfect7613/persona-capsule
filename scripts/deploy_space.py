"""Create or update the official Hugging Face Space without exposing secrets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "build-small-hackathon/persona-capsule"
IGNORED = [
    ".env",
    ".env.*",
    ".git/**",
    ".github/**",
    ".modal/**",
    ".persona-capsule-data/**",
    "**/.persona-capsule-data/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "**/__pycache__/**",
    "artifacts/**",
    "checkpoints/**",
    "outputs/**",
]


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def deployment_plan() -> dict[str, object]:
    return {
        "space_id": SPACE_ID,
        "visibility": "public",
        "sdk": "gradio",
        "hardware": "cpu-basic",
        "oauth": True,
        "secret_names": [
            "HF_TOKEN",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
        ],
        "variable_names": [
            "APP_ENV",
            "ENABLE_BATTLE",
            "ENABLE_CARD",
            "ENABLE_CREATION",
            "ENABLE_DEEP_TRAINING",
            "ENABLE_FUSION",
            "ENABLE_STEERING",
            "ENABLE_VOICE",
            "HF_CAPSULE_REPO_ID",
            "PUBLIC_BASE_URL",
        ],
    }


def deploy(*, dry_run: bool) -> None:
    load_dotenv(ROOT / ".env")
    plan = deployment_plan()
    if dry_run:
        for key, value in plan.items():
            print(f"{key}: {value}")
        return

    if os.environ.get("CONFIRM_CREDENTIALS_ROTATED", "").casefold() != "true":
        raise SystemExit(
            "Deployment refused. Rotate exposed credentials, then set "
            "CONFIRM_CREDENTIALS_ROTATED=true."
        )

    deploy_token = _required("HF_TOKEN")
    service_token = _required("SPACE_HF_SERVICE_TOKEN")
    api = HfApi(token=deploy_token)
    owner = str(api.whoami()["name"])
    data_repo = os.environ.get(
        "HF_CAPSULE_REPO_ID",
        f"{owner}/persona-capsule-private-data",
    ).strip()
    api.create_repo(
        repo_id=data_repo,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        private=False,
        exist_ok=True,
        space_sdk="gradio",
    )

    variables = {
        "APP_ENV": "production",
        "ENABLE_BATTLE": os.environ.get("ENABLE_BATTLE", "true"),
        "ENABLE_CARD": os.environ.get("ENABLE_CARD", "true"),
        "ENABLE_CREATION": os.environ.get("ENABLE_CREATION", "true"),
        # Keep the judged Space focused on the live, working inference path.
        "ENABLE_DEEP_TRAINING": "false",
        "ENABLE_FUSION": os.environ.get("ENABLE_FUSION", "true"),
        "ENABLE_STEERING": os.environ.get("ENABLE_STEERING", "true"),
        "ENABLE_VOICE": os.environ.get("ENABLE_VOICE", "true"),
        "HF_CAPSULE_REPO_ID": data_repo,
        "PUBLIC_BASE_URL": f"https://huggingface.co/spaces/{SPACE_ID}",
    }
    secrets = {
        "HF_TOKEN": service_token,
        "MODAL_TOKEN_ID": _required("MODAL_TOKEN_ID"),
        "MODAL_TOKEN_SECRET": _required("MODAL_TOKEN_SECRET"),
    }
    for key, value in variables.items():
        api.add_space_variable(SPACE_ID, key, value)
    for key, value in secrets.items():
        api.add_space_secret(SPACE_ID, key, value)
    if "ELEVENLABS_API_KEY" in api.get_space_secrets(SPACE_ID):
        api.delete_space_secret(SPACE_ID, "ELEVENLABS_API_KEY")
    commit = api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=ROOT,
        ignore_patterns=IGNORED,
        commit_message="Deploy Persona Capsule hackathon Space",
    )
    api.request_space_hardware(SPACE_ID, "cpu-basic")
    print(f"Deployed {SPACE_ID} at commit {commit.oid}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    deploy(dry_run=arguments.dry_run)
