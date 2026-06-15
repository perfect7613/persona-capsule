from dataclasses import replace
from pathlib import Path

import pytest

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.repository import CapsuleRecord, InMemoryCapsuleRepository
from persona_capsule.voice import (
    CapsuleVoiceService,
    InvalidVoiceAudioError,
    ModalVoxCPMVoiceProvider,
    VoiceClone,
    VoiceError,
    VoiceProviderUnavailableError,
)

OWNER = Principal("hf:owner", "owner", "test")


class FakeVoiceProvider:
    def __init__(self, *, verification_required: bool = False) -> None:
        self.verification_required = verification_required
        self.created: list[tuple[str, tuple[Path, ...]]] = []
        self.synthesized: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.fail_delete = False

    def create_clone(self, *, name: str, audio_paths: tuple[Path, ...]) -> VoiceClone:
        self.created.append((name, audio_paths))
        return VoiceClone("voice-private-123", self.verification_required)

    def synthesize(self, *, voice_id: str, text: str) -> bytes:
        self.synthesized.append((voice_id, text))
        return b"synthetic-mp3"

    def delete_voice(self, voice_id: str) -> None:
        if self.fail_delete:
            raise VoiceProviderUnavailableError("provider unavailable")
        self.deleted.append(voice_id)


def _service(tmp_path: Path, provider: FakeVoiceProvider):
    repository = InMemoryCapsuleRepository()
    library = CapsuleLibrary(repository)
    record = library.save_capsule(
        OWNER,
        CapsuleRecord(
            capsule_id="voice-01",
            owner_id=OWNER.user_id,
            name="Clear Signal",
            status="profile_approved",
        ),
    )
    service = CapsuleVoiceService(library, tmp_path, provider, temporary_hours=6)
    audio = tmp_path / "authorized.wav"
    audio.write_bytes(b"authorized synthetic fixture")
    return library, service, record, audio


def test_modal_voxcpm_adapter_calls_reference_synthesis_and_delete(tmp_path: Path) -> None:
    class RemoteMethod:
        def __init__(self, handler):
            self.handler = handler

        def remote(self, **kwargs):
            return self.handler(**kwargs)

    calls: dict[str, list[dict]] = {"create": [], "synthesize": [], "delete": []}

    class Runtime:
        create_reference = RemoteMethod(
            lambda **kwargs: (
                calls["create"].append(kwargs)
                or {"voice_id": "a" * 32}
            )
        )
        synthesize = RemoteMethod(
            lambda **kwargs: (
                calls["synthesize"].append(kwargs)
                or {"audio_bytes": b"real-audio"}
            )
        )
        delete_reference = RemoteMethod(lambda **kwargs: calls["delete"].append(kwargs))

    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"reference-audio")
    provider = ModalVoxCPMVoiceProvider()
    provider._runtime = lambda: Runtime()

    clone = provider.create_clone(name="Capsule", audio_paths=(sample,))
    audio = provider.synthesize(voice_id=clone.voice_id, text="Hello")
    provider.delete_voice(clone.voice_id)

    assert clone == VoiceClone("a" * 32, False)
    assert audio == b"real-audio"
    assert calls["create"][0]["audio_bytes"] == b"reference-audio"
    assert calls["create"][0]["audio_suffix"] == ".wav"
    assert calls["synthesize"] == [{"voice_id": "a" * 32, "text": "Hello"}]
    assert calls["delete"] == [{"voice_id": "a" * 32}]


def test_clone_requires_consent_and_supported_audio(tmp_path: Path) -> None:
    _, service, record, audio = _service(tmp_path, FakeVoiceProvider())

    with pytest.raises(VoiceError, match="permission"):
        service.create_clone(
            OWNER,
            record.capsule_id,
            [str(audio)],
            "Hello.",
            consented=False,
            retention="temporary",
        )

    invalid = tmp_path / "audio.txt"
    invalid.write_text("not audio")
    with pytest.raises(InvalidVoiceAudioError, match="FLAC"):
        service.create_clone(
            OWNER,
            record.capsule_id,
            [str(invalid)],
            "Hello.",
            consented=True,
            retention="temporary",
        )


def test_create_synthesize_retain_and_delete_real_provider_contract(
    tmp_path: Path,
) -> None:
    provider = FakeVoiceProvider()
    library, service, record, audio = _service(tmp_path, provider)

    created = service.create_clone(
        OWNER,
        record.capsule_id,
        [str(audio)],
        "Small steps, clear signal.",
        consented=True,
        retention="retained",
    )
    speech = service.synthesize(OWNER, record.capsule_id, "Another synthetic line.")

    assert created.audio_path.read_bytes() == b"synthetic-mp3"
    assert speech.read_bytes() == b"synthetic-mp3"
    assert created.record.voice_provider == "openbmb-voxcpm2-modal"
    assert created.record.voice_id == "voice-private-123"
    assert created.record.voice_retention == "retained"
    assert created.record.voice_consent_at
    assert created.record.voice_expires_at == ""
    assert "authorized.wav" not in str(created.record.as_dict())
    assert not audio.exists()
    assert provider.synthesized == [
        ("voice-private-123", "Small steps, clear signal."),
        ("voice-private-123", "Another synthetic line."),
    ]

    deleted = service.delete_voice(OWNER, record.capsule_id)
    again = service.delete_voice(OWNER, record.capsule_id)

    assert provider.deleted == ["voice-private-123"]
    assert deleted.voice_id == ""
    assert deleted.voice_sample_ref == ""
    assert again == deleted
    assert library.get_capsule(OWNER, record.capsule_id).voice_id == ""


def test_verification_required_is_not_saved_as_success(tmp_path: Path) -> None:
    provider = FakeVoiceProvider(verification_required=True)
    library, service, record, audio = _service(tmp_path, provider)

    with pytest.raises(VoiceError, match="verification"):
        service.create_clone(
            OWNER,
            record.capsule_id,
            [str(audio)],
            "Hello.",
            consented=True,
            retention="temporary",
        )

    assert provider.deleted == ["voice-private-123"]
    assert library.get_capsule(OWNER, record.capsule_id).voice_id == ""


def test_delete_failure_records_retryable_cleanup(tmp_path: Path) -> None:
    provider = FakeVoiceProvider()
    library, service, record, audio = _service(tmp_path, provider)
    created = service.create_clone(
        OWNER,
        record.capsule_id,
        [str(audio)],
        "Hello.",
        consented=True,
        retention="temporary",
    )
    provider.fail_delete = True

    with pytest.raises(VoiceProviderUnavailableError):
        service.delete_voice(OWNER, record.capsule_id)

    pending = library.get_capsule(OWNER, created.record.capsule_id)
    assert pending.voice_status == "cleanup_pending"
    assert pending.pending_cleanup_refs == ("voxcpm_reference:voice-private-123",)


def test_expired_temporary_clone_is_cleaned_up(tmp_path: Path) -> None:
    provider = FakeVoiceProvider()
    library, service, record, audio = _service(tmp_path, provider)
    created = service.create_clone(
        OWNER,
        record.capsule_id,
        [str(audio)],
        "Hello.",
        consented=True,
        retention="temporary",
    )
    library.save_capsule(
        OWNER,
        replace(created.record, voice_expires_at="2000-01-01T00:00:00+00:00"),
    )

    assert service.cleanup_expired(OWNER) == (record.capsule_id,)
    assert provider.deleted == ["voice-private-123"]
