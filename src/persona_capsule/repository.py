"""Owner-scoped capsule repository contracts and in-memory implementation."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from threading import RLock

from persona_capsule.profile import ExemplarPair, StyleProfile


class CapsuleNotFoundError(KeyError):
    """Raised without revealing whether a capsule belongs to someone else."""


@dataclass(frozen=True, slots=True)
class CapsuleRecord:
    capsule_id: str
    owner_id: str
    name: str
    status: str = "draft"
    style_profile: StyleProfile | None = None
    exemplar_pairs: tuple[ExemplarPair, ...] = ()
    source_fingerprint: str = ""


class InMemoryCapsuleRepository:
    """Thread-safe repository used until private Hub persistence is introduced."""

    def __init__(self, records: Iterable[CapsuleRecord] = ()) -> None:
        self._lock = RLock()
        self._records = {record.capsule_id: record for record in records}

    def list_for_owner(self, owner_id: str) -> tuple[CapsuleRecord, ...]:
        with self._lock:
            records = [record for record in self._records.values() if record.owner_id == owner_id]
        return tuple(sorted(records, key=lambda record: record.name.casefold()))

    def get_for_owner(self, owner_id: str, capsule_id: str) -> CapsuleRecord:
        with self._lock:
            record = self._records.get(capsule_id)
        if record is None or record.owner_id != owner_id:
            raise CapsuleNotFoundError(capsule_id)
        return record

    def save_for_owner(self, owner_id: str, record: CapsuleRecord) -> CapsuleRecord:
        if record.owner_id != owner_id:
            raise CapsuleNotFoundError(record.capsule_id)
        with self._lock:
            existing = self._records.get(record.capsule_id)
            if existing is not None and existing.owner_id != owner_id:
                raise CapsuleNotFoundError(record.capsule_id)
            self._records[record.capsule_id] = record
        return record

    def rename_for_owner(self, owner_id: str, capsule_id: str, name: str) -> CapsuleRecord:
        record = self.get_for_owner(owner_id, capsule_id)
        updated = replace(record, name=name)
        with self._lock:
            self._records[capsule_id] = updated
        return updated
