"""Tests for visual LoRA training orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from persona_capsule.config import Settings, ensure_data_dirs
from persona_capsule.models.capsule import CapsuleRecord, CreationMode, ExemplarPair
from persona_capsule.models.profile import StyleProfile, StyleTrait
from persona_capsule.services.training_orchestrator import TrainingOrchestrator


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "config" / "generated").mkdir(parents=True)
    return Settings(
        mode="local",
        data_dir=tmp_path / "artifacts",
        repo_root=repo_root,
        host="127.0.0.1",
        port=7860,
        dev_user_id="test-user",
        dev_user_name="Test User",
        hf_token=None,
        modal_token_id=None,
        modal_token_secret=None,
        elevenlabs_api_key=None,
        modal_dataset_path="/dataset/demo",
        modal_model_volume="flux-lora-models",
        flux_base_model="black-forest-labs/FLUX.2-klein-base-4B",
        default_training_steps=500,
    )


@pytest.fixture
def capsule(settings: Settings) -> CapsuleRecord:
    profile = StyleProfile(
        summary="Test capsule",
        tone="warm",
        vocabulary="casual",
        cadence="short",
        traits=[StyleTrait("openness", 0.7, "imaginative", ["bright idea"])],
        palette="gold and navy",
        visual_symbols=["spark"],
    )
    return CapsuleRecord.new(
        owner_id=settings.dev_user_id,
        display_name="Test Capsule",
        profile=profile,
        exemplars=[ExemplarPair("Hello!", "Greetings.")],
        creation_mode=CreationMode.QUICK,
    )


def test_build_training_config(settings: Settings, capsule: CapsuleRecord) -> None:
    ensure_data_dirs(settings)
    orchestrator = TrainingOrchestrator(settings)
    config = orchestrator.build_training_config(capsule, "/dataset/demo")
    process = config["config"]["process"][0]
    assert process["model"]["arch"] == "flux2_klein_4b"
    assert process["datasets"][0]["folder_path"] == "/dataset/demo"
    assert process["trigger_word"].startswith("PSNA")
    assert len(process["sample"]["prompts"]) == 2


def test_submit_dry_run_writes_files(settings: Settings, capsule: CapsuleRecord) -> None:
    ensure_data_dirs(settings)
    orchestrator = TrainingOrchestrator(settings)
    job = orchestrator.submit_visual_lora_training(
        capsule,
        owner_id=settings.dev_user_id,
        dry_run=True,
    )
    assert job.status.value == "submitted"
    assert Path(job.config_path).exists()
    loaded = yaml.safe_load(Path(job.config_path).read_text(encoding="utf-8"))
    assert loaded["config"]["name"] == job.output_name
    saved = json.loads((settings.training_jobs_dir / f"{job.id}.json").read_text(encoding="utf-8"))
    assert saved["capsule_id"] == capsule.id
