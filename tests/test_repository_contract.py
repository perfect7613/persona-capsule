import json
from hashlib import sha256
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from persona_capsule.profile import (
    ExemplarPair,
    StyleDimensions,
    StyleProfile,
)
from persona_capsule.repository import (
    CapsuleNotFoundError,
    CapsuleRecord,
    FileCapsuleRepository,
    HuggingFaceDatasetCapsuleRepository,
    InMemoryCapsuleRepository,
)

ALICE = "hf:alice"
BOB = "hf:bob"


def _profile() -> StyleProfile:
    return StyleProfile(
        summary="Warm, concise, and willing to challenge assumptions.",
        descriptors=("warm", "direct", "analytical"),
        lexical_tendencies=("short transitions", "concrete verbs"),
        sentence_rhythm="Short setup followed by a clear next step.",
        dimensions=StyleDimensions(72, 81, 64, 70, 48, 88, 31),
        evidence=("Private retained sentence.",),
        uncertainty=0.18,
    )


def _record(owner_id: str = ALICE) -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id="abc123",
        owner_id=owner_id,
        name="Clear Signal",
        status="profile_approved",
        style_profile=_profile(),
        exemplar_pairs=(
            ExemplarPair(
                "Good. Test the smallest useful version first.",
                "Test a small useful version first.",
                0,
            ),
        ),
        source_fingerprint="source-v1",
        artifact_refs=("card/interactive.png",),
        pending_cleanup_refs=("voice:temporary-id",),
        voice_provider="elevenlabs",
        voice_id="private-voice-id",
        voice_status="ready",
        voice_retention="retained",
        voice_consent_at="2026-06-13T00:00:00+00:00",
        voice_sample_ref="voice/signature.mp3",
        voice_model_id="eleven_multilingual_v2",
    )


@pytest.mark.parametrize("repository_factory", [InMemoryCapsuleRepository])
def test_repository_contract_is_owner_scoped_and_deletion_is_idempotent(
    repository_factory,
) -> None:
    repository = repository_factory()
    saved = repository.save_for_owner(ALICE, _record())

    assert repository.get_for_owner(ALICE, saved.capsule_id).name == "Clear Signal"
    assert repository.list_for_owner(BOB) == ()
    with pytest.raises(CapsuleNotFoundError):
        repository.get_for_owner(BOB, saved.capsule_id)

    assert repository.delete_for_owner(ALICE, saved.capsule_id) is True
    assert repository.delete_for_owner(ALICE, saved.capsule_id) is False
    with pytest.raises(CapsuleNotFoundError):
        repository.get_public_by_slug("not-published")


def test_file_repository_round_trips_private_and_public_projections(tmp_path: Path) -> None:
    repository = FileCapsuleRepository(tmp_path)
    saved = repository.save_for_owner(ALICE, _record())
    reopened = FileCapsuleRepository(tmp_path).get_for_owner(ALICE, saved.capsule_id)

    assert reopened == saved
    assert reopened.public_projection is not None
    assert reopened.public_projection.summary == reopened.style_profile.summary
    public_payload = reopened.public_projection.as_dict()
    assert "evidence" not in public_payload
    assert "exemplar" not in json.dumps(public_payload)

    owner_namespace = sha256(ALICE.encode()).hexdigest()[:24]
    assert (tmp_path / "records" / owner_namespace / "abc123.json").exists()
    assert "alice" not in str(next((tmp_path / "records").iterdir()))


def test_file_repository_deletes_owned_artifact_directory(tmp_path: Path) -> None:
    repository = FileCapsuleRepository(tmp_path)
    repository.save_for_owner(ALICE, _record())
    owner_namespace = sha256(ALICE.encode()).hexdigest()[:24]
    artifact_dir = tmp_path / "artifacts" / owner_namespace / "abc123"
    nested = artifact_dir / "card"
    nested.mkdir(parents=True)
    (nested / "interactive.png").write_bytes(b"image")

    assert repository.delete_for_owner(ALICE, "abc123") is True
    assert not artifact_dir.exists()


class FakeHubApi:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.created = []
        self.uploaded = []
        self.deleted = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def list_repo_files(self, *args, **kwargs):
        del args, kwargs
        return [str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()]

    def upload_file(self, **kwargs):
        path = self.root / kwargs["path_in_repo"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(kwargs["path_or_fileobj"])
        self.uploaded.append(kwargs)

    def delete_file(self, path_in_repo, **kwargs):
        del kwargs
        (self.root / path_in_repo).unlink()
        self.deleted.append(path_in_repo)


def test_hugging_face_adapter_uses_private_dataset_paths(tmp_path: Path) -> None:
    api = FakeHubApi(tmp_path)

    def download(**kwargs):
        path = tmp_path / kwargs["filename"]
        if not path.exists():
            raise EntryNotFoundError("missing")
        return str(path)

    repository = HuggingFaceDatasetCapsuleRepository(
        "owner/private-capsules",
        "test-token",
        api=api,
        downloader=download,
    )

    saved = repository.save_for_owner(ALICE, _record())
    assert api.created[0]["private"] is True
    assert api.created[0]["repo_type"] == "dataset"
    assert ALICE not in api.uploaded[0]["path_in_repo"]
    assert repository.get_for_owner(ALICE, saved.capsule_id) == saved
    assert repository.delete_for_owner(ALICE, saved.capsule_id) is True
