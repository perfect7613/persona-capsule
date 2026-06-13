"""Capsule persistence layer."""

from __future__ import annotations

import json
from pathlib import Path

from persona_capsule.config import Settings
from persona_capsule.models.capsule import CapsuleRecord, CapsuleState, Visibility


class CapsuleRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.capsules_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, capsule_id: str) -> Path:
        return self.root / f"{capsule_id}.json"

    def save(self, capsule: CapsuleRecord) -> CapsuleRecord:
        capsule.touch()
        self._path(capsule.id).write_text(
            json.dumps(capsule.to_dict(), indent=2),
            encoding="utf-8",
        )
        return capsule

    def get(self, capsule_id: str) -> CapsuleRecord | None:
        path = self._path(capsule_id)
        if not path.exists():
            return None
        return CapsuleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_for_owner(self, owner_id: str) -> list[CapsuleRecord]:
        records: list[CapsuleRecord] = []
        for path in sorted(self.root.glob("*.json")):
            record = CapsuleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if record.owner_id == owner_id and record.state != CapsuleState.DELETED:
                records.append(record)
        records.sort(key=lambda c: c.updated_at, reverse=True)
        return records

    def delete(self, capsule_id: str) -> bool:
        capsule = self.get(capsule_id)
        if capsule is None:
            return False
        capsule.state = CapsuleState.DELETED
        capsule.visibility = Visibility.PRIVATE
        self.save(capsule)
        return True

    def find_by_slug(self, slug: str) -> CapsuleRecord | None:
        for path in self.root.glob("*.json"):
            record = CapsuleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if record.public_slug == slug and record.visibility == Visibility.PUBLIC:
                return record
        return None
