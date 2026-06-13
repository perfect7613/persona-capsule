"""Submit and track FLUX visual LoRA training jobs via ai-toolkit + Modal."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from persona_capsule.config import Settings
from persona_capsule.models.capsule import CapsuleRecord
from persona_capsule.models.training_job import TrainingJobRecord, TrainingJobStatus

DEFAULT_TEMPLATE_NAME = "persona_capsule.template.yaml"


class TrainingOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs_dir = settings.training_jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.generated_configs_dir = settings.generated_configs_dir
        self.generated_configs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save_job(self, job: TrainingJobRecord) -> TrainingJobRecord:
        job.touch()
        self._job_path(job.id).write_text(
            json.dumps(job.to_dict(), indent=2),
            encoding="utf-8",
        )
        return job

    def get_job(self, job_id: str) -> TrainingJobRecord | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return TrainingJobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_jobs_for_capsule(self, capsule_id: str) -> list[TrainingJobRecord]:
        jobs: list[TrainingJobRecord] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            job = TrainingJobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if job.capsule_id == capsule_id:
                jobs.append(job)
        jobs.sort(key=lambda j: j.updated_at, reverse=True)
        return jobs

    @staticmethod
    def _sanitize_trigger_word(capsule: CapsuleRecord) -> str:
        token = re.sub(r"[^A-Za-z0-9]", "", capsule.display_name)[:6].upper()
        suffix = capsule.id.replace("-", "")[:4].upper()
        return f"PSNA{token or 'CAPS'}{suffix}"

    def _build_sample_prompts(self, capsule: CapsuleRecord, trigger_word: str) -> list[str]:
        profile = capsule.profile
        symbols = ", ".join(profile.visual_symbols[:3]) or "collectible card motifs"
        return [
            (
                f"{trigger_word}. Persona Capsule card art, {profile.tone}, "
                f"{profile.palette}, {profile.visual_energy} energy, {symbols}"
            ),
            (
                f"{trigger_word}. Portrait collectible card, {profile.vocabulary} style, "
                f"soft cinematic lighting, {profile.cadence}"
            ),
        ]

    def build_training_config(
        self,
        capsule: CapsuleRecord,
        dataset_folder_path: str,
        *,
        steps: int | None = None,
        trigger_word: str | None = None,
    ) -> dict:
        trigger = trigger_word or self._sanitize_trigger_word(capsule)
        output_name = f"persona_capsule_{capsule.id[:8]}"
        train_steps = steps or self.settings.default_training_steps
        return {
            "job": "extension",
            "config": {
                "name": output_name,
                "training_folder": "/output",
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": "/output",
                        "device": "cuda:0",
                        "trigger_word": trigger,
                        "save": {
                            "save_every": max(250, train_steps // 4),
                            "save_format": "safetensors",
                        },
                        "datasets": [
                            {
                                "folder_path": dataset_folder_path,
                                "caption_ext": "txt",
                                "resolution": [1024],
                            }
                        ],
                        "train": {
                            "batch_size": 1,
                            "steps": train_steps,
                            "gradient_accumulation_steps": 1,
                            "lr": 1e-4,
                            "optimizer": "adamw8bit",
                            "noise_scheduler": "flowmatch",
                            "dtype": "bf16",
                            "gradient_checkpointing": True,
                        },
                        "model": {
                            "name_or_path": self.settings.flux_base_model,
                            "arch": "flux2_klein_4b",
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": max(250, train_steps // 4),
                            "sample_steps": 4,
                            "prompts": self._build_sample_prompts(capsule, trigger),
                        },
                    }
                ],
            },
        }

    def write_config_files(
        self,
        capsule: CapsuleRecord,
        dataset_folder_path: str,
        *,
        steps: int | None = None,
        trigger_word: str | None = None,
    ) -> tuple[Path, Path, dict, str]:
        config_dict = self.build_training_config(
            capsule,
            dataset_folder_path,
            steps=steps,
            trigger_word=trigger_word,
        )
        output_name = config_dict["config"]["name"]
        trigger = config_dict["config"]["process"][0]["trigger_word"]

        local_config = self.generated_configs_dir / f"{output_name}.yaml"
        local_config.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")

        modal_config = f"/root/ai-toolkit/{local_config.relative_to(self.settings.repo_root).as_posix()}"
        return local_config, Path(modal_config), config_dict, trigger

    def submit_visual_lora_training(
        self,
        capsule: CapsuleRecord,
        owner_id: str,
        dataset_folder_path: str | None = None,
        *,
        steps: int | None = None,
        trigger_word: str | None = None,
        dry_run: bool = False,
    ) -> TrainingJobRecord:
        dataset_path = dataset_folder_path or self.settings.modal_dataset_path
        if not dataset_path:
            raise ValueError(
                "Dataset folder path is required. Set MODAL_DATASET_PATH or pass dataset_folder_path."
            )

        local_config, modal_config, _, trigger = self.write_config_files(
            capsule,
            dataset_path,
            steps=steps,
            trigger_word=trigger_word,
        )
        output_name = local_config.stem
        job = TrainingJobRecord.new(
            capsule_id=capsule.id,
            owner_id=owner_id,
            config_path=str(local_config),
            modal_config_path=modal_config.as_posix(),
            trigger_word=trigger,
            dataset_folder_path=dataset_path,
            output_name=output_name,
        )

        if dry_run or self.settings.use_mock_providers:
            job.status = TrainingJobStatus.SUBMITTED
            job.submitted_command = (
                f"modal run {self.settings.repo_root / 'run_modal.py'} --config {modal_config.as_posix()}"
            )
            job.error_message = None
            return self.save_job(job)

        if not self.settings.modal_token_id or not self.settings.modal_token_secret:
            raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required to submit training.")

        command = [
            sys.executable,
            "-m",
            "modal",
            "run",
            str(self.settings.repo_root / "run_modal.py"),
            "--config",
            modal_config.as_posix(),
        ]
        job.submitted_command = " ".join(command)
        job.status = TrainingJobStatus.SUBMITTED
        self.save_job(job)

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.settings.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                job.status = TrainingJobStatus.COMPLETED
                job.output_lora_path = (
                    f"modal://flux-lora-models/{output_name}"
                )
                job.error_message = None
            else:
                job.status = TrainingJobStatus.FAILED
                job.error_message = (completed.stderr or completed.stdout or "Modal run failed.")[:4000]
        except FileNotFoundError as exc:
            job.status = TrainingJobStatus.FAILED
            job.error_message = f"Modal CLI not found: {exc}"

        return self.save_job(job)

    def attach_latest_job_to_capsule(self, capsule: CapsuleRecord) -> CapsuleRecord:
        jobs = self.list_jobs_for_capsule(capsule.id)
        if not jobs:
            return capsule
        latest = jobs[0]
        capsule.visual_lora_job_id = latest.id
        capsule.visual_trigger_word = latest.trigger_word
        if latest.output_lora_path:
            capsule.visual_lora_path = latest.output_lora_path
        capsule.visual_training_status = latest.status.value
        return capsule
