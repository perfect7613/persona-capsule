"""Visual LoRA training job records."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TrainingJobStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TrainingJobRecord:
    id: str
    capsule_id: str
    owner_id: str
    status: TrainingJobStatus
    config_path: str
    modal_config_path: str
    trigger_word: str
    dataset_folder_path: str
    output_name: str
    submitted_command: str | None = None
    modal_run_id: str | None = None
    error_message: str | None = None
    output_lora_path: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def new(
        cls,
        capsule_id: str,
        owner_id: str,
        config_path: str,
        modal_config_path: str,
        trigger_word: str,
        dataset_folder_path: str,
        output_name: str,
    ) -> TrainingJobRecord:
        return cls(
            id=str(uuid.uuid4()),
            capsule_id=capsule_id,
            owner_id=owner_id,
            status=TrainingJobStatus.PENDING,
            config_path=config_path,
            modal_config_path=modal_config_path,
            trigger_word=trigger_word,
            dataset_folder_path=dataset_folder_path,
            output_name=output_name,
        )

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingJobRecord:
        payload = dict(data)
        payload["status"] = TrainingJobStatus(payload["status"])
        return cls(**payload)
