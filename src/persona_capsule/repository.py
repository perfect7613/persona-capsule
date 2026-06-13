"""Owner-scoped capsule repository contracts and persistence adapters."""

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any, Protocol

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from persona_capsule.profile import ExemplarPair, StyleProfile

CAPSULE_SCHEMA_VERSION = "persona-capsule-v1"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class CapsuleNotFoundError(KeyError):
    """Raised without revealing whether a capsule belongs to someone else."""


@dataclass(frozen=True, slots=True)
class CapsulePublicProjection:
    name: str
    summary: str
    descriptors: tuple[str, ...]
    dimensions: dict[str, float]

    @classmethod
    def from_profile(cls, name: str, profile: StyleProfile) -> "CapsulePublicProjection":
        return cls(
            name=name,
            summary=profile.summary,
            descriptors=profile.descriptors,
            dimensions=profile.dimensions.as_dict(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "descriptors": list(self.descriptors),
            "dimensions": dict(self.dimensions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CapsulePublicProjection":
        return cls(
            name=str(payload["name"]),
            summary=str(payload["summary"]),
            descriptors=tuple(map(str, payload.get("descriptors", ()))),
            dimensions={
                str(key): float(value) for key, value in dict(payload.get("dimensions", {})).items()
            },
        )


@dataclass(frozen=True, slots=True)
class CapsuleRecord:
    capsule_id: str
    owner_id: str
    name: str
    status: str = "draft"
    style_profile: StyleProfile | None = None
    exemplar_pairs: tuple[ExemplarPair, ...] = ()
    source_fingerprint: str = ""
    public_projection: CapsulePublicProjection | None = None
    artifact_refs: tuple[str, ...] = ()
    pending_cleanup_refs: tuple[str, ...] = ()
    card_image_ref: str = ""
    social_image_ref: str = ""
    card_seed: int | None = None
    card_prompt_hash: str = ""
    card_provider: str = ""
    card_model_id: str = ""
    card_model_revision: str = ""
    voice_provider: str = ""
    voice_id: str = ""
    voice_status: str = ""
    voice_retention: str = ""
    voice_consent_at: str = ""
    voice_expires_at: str = ""
    voice_sample_ref: str = ""
    voice_model_id: str = ""
    is_published: bool = False
    public_slug: str = ""
    published_projection: dict[str, Any] | None = None
    published_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    def canonicalized(self) -> "CapsuleRecord":
        now = _now_iso()
        projection = self.public_projection
        if projection is None and self.style_profile is not None:
            projection = CapsulePublicProjection.from_profile(self.name, self.style_profile)
        return replace(
            self,
            public_projection=projection,
            created_at=self.created_at or now,
            updated_at=now,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPSULE_SCHEMA_VERSION,
            "capsule_id": self.capsule_id,
            "owner_id": self.owner_id,
            "name": self.name,
            "status": self.status,
            "style_profile": (
                self.style_profile.as_dict(include_private_evidence=True)
                if self.style_profile
                else None
            ),
            "exemplar_pairs": [pair.as_dict() for pair in self.exemplar_pairs],
            "source_fingerprint": self.source_fingerprint,
            "public_projection": (
                self.public_projection.as_dict() if self.public_projection else None
            ),
            "artifact_refs": list(self.artifact_refs),
            "pending_cleanup_refs": list(self.pending_cleanup_refs),
            "card_image_ref": self.card_image_ref,
            "social_image_ref": self.social_image_ref,
            "card_seed": self.card_seed,
            "card_prompt_hash": self.card_prompt_hash,
            "card_provider": self.card_provider,
            "card_model_id": self.card_model_id,
            "card_model_revision": self.card_model_revision,
            "voice_provider": self.voice_provider,
            "voice_id": self.voice_id,
            "voice_status": self.voice_status,
            "voice_retention": self.voice_retention,
            "voice_consent_at": self.voice_consent_at,
            "voice_expires_at": self.voice_expires_at,
            "voice_sample_ref": self.voice_sample_ref,
            "voice_model_id": self.voice_model_id,
            "is_published": self.is_published,
            "public_slug": self.public_slug,
            "published_projection": self.published_projection,
            "published_at": self.published_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CapsuleRecord":
        if payload.get("schema_version") != CAPSULE_SCHEMA_VERSION:
            raise ValueError("Unsupported capsule schema version.")
        profile_payload = payload.get("style_profile")
        projection_payload = payload.get("public_projection")
        return cls(
            capsule_id=str(payload["capsule_id"]),
            owner_id=str(payload["owner_id"]),
            name=str(payload["name"]),
            status=str(payload.get("status", "draft")),
            style_profile=(
                StyleProfile.from_dict(profile_payload) if profile_payload is not None else None
            ),
            exemplar_pairs=tuple(
                ExemplarPair.from_dict(pair) for pair in payload.get("exemplar_pairs", ())
            ),
            source_fingerprint=str(payload.get("source_fingerprint", "")),
            public_projection=(
                CapsulePublicProjection.from_dict(projection_payload)
                if projection_payload is not None
                else None
            ),
            artifact_refs=tuple(map(str, payload.get("artifact_refs", ()))),
            pending_cleanup_refs=tuple(map(str, payload.get("pending_cleanup_refs", ()))),
            card_image_ref=str(payload.get("card_image_ref", "")),
            social_image_ref=str(payload.get("social_image_ref", "")),
            card_seed=(int(payload["card_seed"]) if payload.get("card_seed") is not None else None),
            card_prompt_hash=str(payload.get("card_prompt_hash", "")),
            card_provider=str(payload.get("card_provider", "")),
            card_model_id=str(payload.get("card_model_id", "")),
            card_model_revision=str(payload.get("card_model_revision", "")),
            voice_provider=str(payload.get("voice_provider", "")),
            voice_id=str(payload.get("voice_id", "")),
            voice_status=str(payload.get("voice_status", "")),
            voice_retention=str(payload.get("voice_retention", "")),
            voice_consent_at=str(payload.get("voice_consent_at", "")),
            voice_expires_at=str(payload.get("voice_expires_at", "")),
            voice_sample_ref=str(payload.get("voice_sample_ref", "")),
            voice_model_id=str(payload.get("voice_model_id", "")),
            is_published=bool(payload.get("is_published", False)),
            public_slug=str(payload.get("public_slug", "")),
            published_projection=(
                dict(payload["published_projection"])
                if payload.get("published_projection") is not None
                else None
            ),
            published_at=str(payload.get("published_at", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )


class CapsuleRepository(Protocol):
    def list_for_owner(self, owner_id: str) -> tuple[CapsuleRecord, ...]: ...

    def get_for_owner(self, owner_id: str, capsule_id: str) -> CapsuleRecord: ...

    def save_for_owner(self, owner_id: str, record: CapsuleRecord) -> CapsuleRecord: ...

    def rename_for_owner(
        self,
        owner_id: str,
        capsule_id: str,
        name: str,
    ) -> CapsuleRecord: ...

    def delete_for_owner(self, owner_id: str, capsule_id: str) -> bool: ...

    def get_public_by_slug(self, slug: str) -> CapsuleRecord: ...


class InMemoryCapsuleRepository:
    """Thread-safe repository for tests and explicitly ephemeral sessions."""

    def __init__(self, records: Iterable[CapsuleRecord] = ()) -> None:
        self._lock = RLock()
        self._records = {record.capsule_id: record.canonicalized() for record in records}

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
            canonical = record.canonicalized()
            self._records[record.capsule_id] = canonical
        return canonical

    def rename_for_owner(self, owner_id: str, capsule_id: str, name: str) -> CapsuleRecord:
        record = self.get_for_owner(owner_id, capsule_id)
        projection = record.public_projection
        if projection is not None:
            projection = replace(projection, name=name)
        return self.save_for_owner(
            owner_id,
            replace(record, name=name, public_projection=projection),
        )

    def delete_for_owner(self, owner_id: str, capsule_id: str) -> bool:
        with self._lock:
            record = self._records.get(capsule_id)
            if record is None:
                return False
            if record.owner_id != owner_id:
                raise CapsuleNotFoundError(capsule_id)
            self._records.pop(capsule_id)
        return True

    def get_public_by_slug(self, slug: str) -> CapsuleRecord:
        with self._lock:
            for record in self._records.values():
                if record.is_published and record.public_slug == slug:
                    return record
        raise CapsuleNotFoundError(slug)


class FileCapsuleRepository:
    """Atomic local JSON persistence with owner-hashed storage paths."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._records_root = self._root / "records"
        self._artifacts_root = self._root / "artifacts"
        self._lock = RLock()

    @staticmethod
    def _owner_namespace(owner_id: str) -> str:
        return sha256(owner_id.encode()).hexdigest()[:24]

    def _owner_dir(self, owner_id: str) -> Path:
        return self._records_root / self._owner_namespace(owner_id)

    def _record_path(self, owner_id: str, capsule_id: str) -> Path:
        if not capsule_id or any(character not in "0123456789abcdef-" for character in capsule_id):
            raise CapsuleNotFoundError(capsule_id)
        return self._owner_dir(owner_id) / f"{capsule_id}.json"

    def list_for_owner(self, owner_id: str) -> tuple[CapsuleRecord, ...]:
        owner_dir = self._owner_dir(owner_id)
        if not owner_dir.exists():
            return ()
        records = [
            CapsuleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in owner_dir.glob("*.json")
        ]
        return tuple(sorted(records, key=lambda record: record.name.casefold()))

    def get_for_owner(self, owner_id: str, capsule_id: str) -> CapsuleRecord:
        path = self._record_path(owner_id, capsule_id)
        try:
            record = CapsuleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError as exc:
            raise CapsuleNotFoundError(capsule_id) from exc
        if record.owner_id != owner_id:
            raise CapsuleNotFoundError(capsule_id)
        return record

    def save_for_owner(self, owner_id: str, record: CapsuleRecord) -> CapsuleRecord:
        if record.owner_id != owner_id:
            raise CapsuleNotFoundError(record.capsule_id)
        canonical = record.canonicalized()
        path = self._record_path(owner_id, canonical.capsule_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            canonical.as_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        with (
            self._lock,
            NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
            ) as temporary,
        ):
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        path.chmod(0o600)
        return canonical

    def rename_for_owner(self, owner_id: str, capsule_id: str, name: str) -> CapsuleRecord:
        record = self.get_for_owner(owner_id, capsule_id)
        projection = record.public_projection
        if projection is not None:
            projection = replace(projection, name=name)
        return self.save_for_owner(
            owner_id,
            replace(record, name=name, public_projection=projection),
        )

    def delete_for_owner(self, owner_id: str, capsule_id: str) -> bool:
        path = self._record_path(owner_id, capsule_id)
        try:
            record = self.get_for_owner(owner_id, capsule_id)
        except CapsuleNotFoundError:
            return False
        path.unlink(missing_ok=True)
        artifact_dir = self._artifacts_root / self._owner_namespace(owner_id) / record.capsule_id
        if artifact_dir.exists():
            for artifact in sorted(
                artifact_dir.rglob("*"),
                key=lambda candidate: len(candidate.parts),
                reverse=True,
            ):
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
                elif artifact.is_dir():
                    artifact.rmdir()
            artifact_dir.rmdir()
        return True

    def get_public_by_slug(self, slug: str) -> CapsuleRecord:
        if not self._records_root.exists():
            raise CapsuleNotFoundError(slug)
        for path in self._records_root.glob("*/*.json"):
            record = CapsuleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if record.is_published and record.public_slug == slug:
                return record
        raise CapsuleNotFoundError(slug)


class HubApiLike(Protocol):
    def create_repo(self, **kwargs: Any) -> Any: ...

    def list_repo_files(self, *args: Any, **kwargs: Any) -> list[str]: ...

    def upload_file(self, **kwargs: Any) -> Any: ...

    def delete_file(self, *args: Any, **kwargs: Any) -> Any: ...


class HuggingFaceDatasetCapsuleRepository:
    """Private dataset-backed adapter used by the deployed Space."""

    def __init__(
        self,
        repo_id: str,
        token: str,
        *,
        api: HubApiLike | None = None,
        downloader: Callable[..., str] = hf_hub_download,
    ) -> None:
        if not repo_id or not token:
            raise ValueError("Hugging Face capsule storage requires repo ID and token.")
        self._repo_id = repo_id
        self._token = token
        self._api = api or HfApi(token=token)
        self._downloader = downloader
        self._repo_ready = False

    @staticmethod
    def _owner_namespace(owner_id: str) -> str:
        return sha256(owner_id.encode()).hexdigest()[:24]

    def _path(self, owner_id: str, capsule_id: str) -> str:
        return f"capsules/{self._owner_namespace(owner_id)}/{capsule_id}.json"

    def _ensure_repo(self) -> None:
        if self._repo_ready:
            return
        self._api.create_repo(
            repo_id=self._repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token=self._token,
        )
        self._repo_ready = True

    def _files(self) -> list[str]:
        try:
            return self._api.list_repo_files(
                self._repo_id,
                repo_type="dataset",
                token=self._token,
            )
        except RepositoryNotFoundError:
            return []

    def list_for_owner(self, owner_id: str) -> tuple[CapsuleRecord, ...]:
        prefix = f"capsules/{self._owner_namespace(owner_id)}/"
        records = []
        for path in self._files():
            if path.startswith(prefix) and path.endswith(".json"):
                records.append(self._download_record(path))
        return tuple(sorted(records, key=lambda record: record.name.casefold()))

    def _download_record(self, path: str) -> CapsuleRecord:
        local_path = self._downloader(
            repo_id=self._repo_id,
            filename=path,
            repo_type="dataset",
            token=self._token,
        )
        return CapsuleRecord.from_dict(json.loads(Path(local_path).read_text(encoding="utf-8")))

    def get_for_owner(self, owner_id: str, capsule_id: str) -> CapsuleRecord:
        path = self._path(owner_id, capsule_id)
        try:
            record = self._download_record(path)
        except (EntryNotFoundError, RepositoryNotFoundError) as exc:
            raise CapsuleNotFoundError(capsule_id) from exc
        if record.owner_id != owner_id:
            raise CapsuleNotFoundError(capsule_id)
        return record

    def save_for_owner(self, owner_id: str, record: CapsuleRecord) -> CapsuleRecord:
        if record.owner_id != owner_id:
            raise CapsuleNotFoundError(record.capsule_id)
        canonical = record.canonicalized()
        self._ensure_repo()
        self._api.upload_file(
            path_or_fileobj=json.dumps(
                canonical.as_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            ).encode(),
            path_in_repo=self._path(owner_id, canonical.capsule_id),
            repo_id=self._repo_id,
            repo_type="dataset",
            token=self._token,
            commit_message=f"Save capsule {canonical.capsule_id}",
        )
        return canonical

    def rename_for_owner(self, owner_id: str, capsule_id: str, name: str) -> CapsuleRecord:
        record = self.get_for_owner(owner_id, capsule_id)
        projection = record.public_projection
        if projection is not None:
            projection = replace(projection, name=name)
        return self.save_for_owner(
            owner_id,
            replace(record, name=name, public_projection=projection),
        )

    def delete_for_owner(self, owner_id: str, capsule_id: str) -> bool:
        path = self._path(owner_id, capsule_id)
        if path not in self._files():
            return False
        self._api.delete_file(
            path,
            repo_id=self._repo_id,
            repo_type="dataset",
            token=self._token,
            commit_message=f"Delete capsule {capsule_id}",
        )
        return True

    def get_public_by_slug(self, slug: str) -> CapsuleRecord:
        for path in self._files():
            if not path.startswith("capsules/") or not path.endswith(".json"):
                continue
            record = self._download_record(path)
            if record.is_published and record.public_slug == slug:
                return record
        raise CapsuleNotFoundError(slug)
