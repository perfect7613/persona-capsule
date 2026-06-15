"""Consent-gated OpenBMB VoxCPM2 voice cloning and speech lifecycle."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.repository import CapsuleRecord

VOICE_PROVIDER_ID = "openbmb-voxcpm2-modal"
VOICE_MODEL_ID = "openbmb/VoxCPM2"
VOICE_MODAL_APP_NAME = "persona-capsule-voxcpm"
VOICE_MODAL_CLASS_NAME = "VoxCPMVoiceRuntime"
SUPPORTED_AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


class VoiceError(RuntimeError):
    """Base error safe to show in the creator UI."""


class VoiceProviderUnavailableError(VoiceError):
    """Raised when the VoxCPM2 Modal runtime cannot be reached."""


class VoiceQuotaError(VoiceError):
    """Raised when the provider account has insufficient quota."""


class InvalidVoiceAudioError(VoiceError):
    """Raised when source audio cannot be used for cloning."""


@dataclass(frozen=True, slots=True)
class VoiceClone:
    voice_id: str
    requires_verification: bool


class VoiceProvider(Protocol):
    def create_clone(
        self,
        *,
        name: str,
        audio_paths: tuple[Path, ...],
    ) -> VoiceClone: ...

    def synthesize(self, *, voice_id: str, text: str) -> bytes: ...

    def delete_voice(self, voice_id: str) -> None: ...


def _raise_modal_error(error: Exception) -> None:
    text = str(error).casefold()
    if any(marker in text for marker in ("audio", "decode", "format")):
        raise InvalidVoiceAudioError(
            "VoxCPM2 could not use this recording. Upload a clear supported audio file."
        ) from error
    if any(marker in text for marker in ("quota", "rate limit", "resource exhausted")):
        raise VoiceQuotaError(
            "The VoxCPM2 runtime quota was reached. Try again when Modal capacity is available."
        ) from error
    raise VoiceProviderUnavailableError(
        "The VoxCPM2 voice runtime is temporarily unavailable. The capsule itself is still safe."
    ) from error


class ModalVoxCPMVoiceProvider:
    """Production provider backed by OpenBMB VoxCPM2 on Modal."""

    @staticmethod
    def _runtime():
        import modal

        return modal.Cls.from_name(VOICE_MODAL_APP_NAME, VOICE_MODAL_CLASS_NAME)()

    def create_clone(
        self,
        *,
        name: str,
        audio_paths: tuple[Path, ...],
    ) -> VoiceClone:
        if len(audio_paths) != 1:
            raise InvalidVoiceAudioError("Upload one clear reference recording for VoxCPM2.")
        try:
            path = audio_paths[0]
            response = self._runtime().create_reference.remote(
                audio_bytes=path.read_bytes(),
                audio_suffix=path.suffix.casefold(),
            )
        except Exception as error:
            _raise_modal_error(error)
        return VoiceClone(
            voice_id=str(response["voice_id"]),
            requires_verification=False,
        )

    def synthesize(self, *, voice_id: str, text: str) -> bytes:
        try:
            response = self._runtime().synthesize.remote(
                voice_id=voice_id,
                text=text,
            )
        except Exception as error:
            _raise_modal_error(error)
        return bytes(response["audio_bytes"])

    def delete_voice(self, voice_id: str) -> None:
        try:
            self._runtime().delete_reference.remote(voice_id=voice_id)
        except Exception as error:
            _raise_modal_error(error)


@dataclass(frozen=True, slots=True)
class VoiceCreationResult:
    record: CapsuleRecord
    audio_path: Path


class CapsuleVoiceService:
    def __init__(
        self,
        capsule_library: CapsuleLibrary,
        artifact_root: str | Path,
        provider: VoiceProvider | None,
        *,
        temporary_hours: int = 24,
    ) -> None:
        self._capsule_library = capsule_library
        self._artifact_root = Path(artifact_root)
        self._provider = provider
        self._temporary_hours = max(1, temporary_hours)

    def _provider_or_raise(self) -> VoiceProvider:
        if self._provider is None:
            raise VoiceProviderUnavailableError(
                "VoxCPM2 voice cloning is unavailable until the Modal runtime is configured."
            )
        return self._provider

    @staticmethod
    def _validated_audio_paths(audio_paths: list[str] | tuple[str, ...]) -> tuple[Path, ...]:
        paths = tuple(Path(value) for value in audio_paths if str(value).strip())
        if not paths:
            raise InvalidVoiceAudioError("Upload at least one authorized voice recording.")
        for path in paths:
            if not path.is_file():
                raise InvalidVoiceAudioError("An uploaded voice recording is no longer available.")
            if path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
                raise InvalidVoiceAudioError(
                    "Use FLAC, MP3, OGG, or WAV audio."
                )
            if path.stat().st_size <= 0 or path.stat().st_size > MAX_AUDIO_BYTES:
                raise InvalidVoiceAudioError("Each recording must be between 1 byte and 25 MB.")
        return paths

    def create_clone(
        self,
        principal: Principal | None,
        capsule_id: str,
        audio_paths: list[str] | tuple[str, ...],
        signature_text: str,
        *,
        consented: bool,
        retention: str,
    ) -> VoiceCreationResult:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        if not consented:
            raise VoiceError("Confirm that you own this voice or have permission to clone it.")
        if retention not in {"temporary", "retained"}:
            raise VoiceError("Choose temporary or retained voice lifecycle.")
        text = signature_text.strip()
        if not text or len(text) > 500:
            raise VoiceError("Provide a signature line between 1 and 500 characters.")
        record = self._capsule_library.get_capsule(principal, capsule_id)
        if record.voice_id:
            raise VoiceError("Delete the existing capsule voice before creating another.")
        paths = self._validated_audio_paths(audio_paths)
        provider = self._provider_or_raise()
        try:
            clone = provider.create_clone(
                name=f"{record.name} Persona Capsule",
                audio_paths=paths,
            )
        finally:
            for path in paths:
                path.unlink(missing_ok=True)
        if clone.requires_verification:
            try:
                provider.delete_voice(clone.voice_id)
            except VoiceError:
                pass
            raise VoiceError("The voice provider requires additional ownership verification.")

        try:
            audio = provider.synthesize(voice_id=clone.voice_id, text=text)
            if not audio:
                raise VoiceProviderUnavailableError("VoxCPM2 returned an empty audio response.")
        except Exception:
            try:
                provider.delete_voice(clone.voice_id)
            except VoiceError:
                pass
            raise

        owner_namespace = sha256(principal.user_id.encode()).hexdigest()[:24]
        output_dir = self._artifact_root / "artifacts" / owner_namespace / record.capsule_id
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "voice-signature.wav"
        audio_path.write_bytes(audio)
        audio_path.chmod(0o600)
        now = _now()
        expires_at = (
            (now + timedelta(hours=self._temporary_hours)).isoformat()
            if retention == "temporary"
            else ""
        )
        reference = "voice/voice-signature.wav"
        updated = self._capsule_library.save_capsule(
            principal,
            replace(
                record,
                artifact_refs=tuple(
                    item for item in record.artifact_refs if not item.startswith("voice/")
                )
                + (reference,),
                voice_provider=VOICE_PROVIDER_ID,
                voice_id=clone.voice_id,
                voice_status="ready",
                voice_retention=retention,
                voice_consent_at=now.isoformat(),
                voice_expires_at=expires_at,
                voice_sample_ref=reference,
                voice_model_id=VOICE_MODEL_ID,
                pending_cleanup_refs=tuple(
                    item
                    for item in record.pending_cleanup_refs
                    if not item.startswith("voxcpm_reference:")
                ),
            ),
        )
        return VoiceCreationResult(record=updated, audio_path=audio_path)

    def synthesize(
        self,
        principal: Principal | None,
        capsule_id: str,
        text: str,
    ) -> Path:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        clean_text = text.strip()
        if not clean_text or len(clean_text) > 1000:
            raise VoiceError("Speech text must be between 1 and 1000 characters.")
        record = self._capsule_library.get_capsule(principal, capsule_id)
        if not record.voice_id or record.voice_status != "ready":
            raise VoiceError("Create an available capsule voice before synthesizing speech.")
        audio = self._provider_or_raise().synthesize(
            voice_id=record.voice_id,
            text=clean_text,
        )
        owner_namespace = sha256(principal.user_id.encode()).hexdigest()[:24]
        output_dir = self._artifact_root / "artifacts" / owner_namespace / record.capsule_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "voice-latest.wav"
        output_path.write_bytes(audio)
        output_path.chmod(0o600)
        return output_path

    def delete_voice(
        self,
        principal: Principal | None,
        capsule_id: str,
    ) -> CapsuleRecord:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        record = self._capsule_library.get_capsule(principal, capsule_id)
        if not record.voice_id:
            return record
        cleanup_ref = f"voxcpm_reference:{record.voice_id}"
        try:
            self._provider_or_raise().delete_voice(record.voice_id)
        except VoiceError:
            pending = tuple(dict.fromkeys(record.pending_cleanup_refs + (cleanup_ref,)))
            self._capsule_library.save_capsule(
                principal,
                replace(
                    record,
                    voice_status="cleanup_pending",
                    pending_cleanup_refs=pending,
                ),
            )
            raise

        for reference in (record.voice_sample_ref, "voice/voice-latest.wav"):
            if not reference:
                continue
            owner_namespace = sha256(principal.user_id.encode()).hexdigest()[:24]
            path = (
                self._artifact_root
                / "artifacts"
                / owner_namespace
                / record.capsule_id
                / Path(reference).name
            )
            path.unlink(missing_ok=True)
        return self._capsule_library.save_capsule(
            principal,
            replace(
                record,
                artifact_refs=tuple(
                    item for item in record.artifact_refs if not item.startswith("voice/")
                ),
                pending_cleanup_refs=tuple(
                    item
                    for item in record.pending_cleanup_refs
                    if not item.startswith("voxcpm_reference:")
                ),
                voice_provider="",
                voice_id="",
                voice_status="",
                voice_retention="",
                voice_consent_at="",
                voice_expires_at="",
                voice_sample_ref="",
                voice_model_id="",
            ),
        )

    def cleanup_expired(
        self,
        principal: Principal | None,
    ) -> tuple[str, ...]:
        if principal is None:
            return ()
        cleaned: list[str] = []
        now = _now()
        for record in self._capsule_library.list_capsules(principal):
            if record.voice_retention != "temporary" or not record.voice_expires_at:
                continue
            try:
                expires_at = datetime.fromisoformat(record.voice_expires_at)
            except ValueError:
                continue
            if expires_at <= now:
                try:
                    self.delete_voice(principal, record.capsule_id)
                except VoiceError:
                    continue
                cleaned.append(record.capsule_id)
        return tuple(cleaned)
