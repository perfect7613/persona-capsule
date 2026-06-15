"""Modal OpenBMB VoxCPM2 runtime for consented voice cloning and speech."""

import io
import re
from pathlib import Path
from uuid import uuid4

import modal

APP_NAME = "persona-capsule-voxcpm"
MODEL_ID = "openbmb/VoxCPM2"
MODEL_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d"
MODEL_CACHE_PATH = "/models"
REFERENCE_CACHE_PATH = "/voice-references"

model_volume = modal.Volume.from_name(
    "persona-capsule-voxcpm-models",
    create_if_missing=True,
)
reference_volume = modal.Volume.from_name(
    "persona-capsule-voice-references",
    create_if_missing=True,
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface-hub[hf-xet]==0.36.0",
        "soundfile==0.13.1",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "voxcpm==2.0.3",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)
app = modal.App(APP_NAME)


def _safe_reference_id(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid VoxCPM2 reference identifier.")
    return value


@app.cls(
    image=image,
    gpu="A10G",
    timeout=20 * 60,
    scaledown_window=5 * 60,
    max_containers=1,
    volumes={
        MODEL_CACHE_PATH: model_volume,
        REFERENCE_CACHE_PATH: reference_volume,
    },
    secrets=[modal.Secret.from_name("persona-capsule-huggingface")],
)
class VoxCPMVoiceRuntime:
    @modal.enter()
    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from voxcpm import VoxCPM

        model_path = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=f"{MODEL_CACHE_PATH}/VoxCPM2",
        )
        model_volume.commit()
        self.model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=False,
            optimize=False,
            device="cuda",
        )

    @modal.method()
    def create_reference(
        self,
        *,
        audio_bytes: bytes,
        audio_suffix: str,
    ) -> dict:
        if not audio_bytes or len(audio_bytes) > 25 * 1024 * 1024:
            raise ValueError("Reference audio must be between 1 byte and 25 MB.")
        if audio_suffix not in {".flac", ".mp3", ".ogg", ".wav"}:
            raise ValueError("Unsupported reference audio format.")

        import librosa
        import soundfile as sf

        reference_id = uuid4().hex
        output_path = Path(REFERENCE_CACHE_PATH) / f"{reference_id}.wav"
        try:
            samples, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if sample_rate != 16000:
                samples = librosa.resample(
                    samples,
                    orig_sr=sample_rate,
                    target_sr=16000,
                )
            sf.write(output_path, samples, 16000, format="WAV")
        except Exception as error:
            output_path.unlink(missing_ok=True)
            raise ValueError("VoxCPM2 could not decode the reference audio.") from error
        reference_volume.commit()
        return {
            "voice_id": reference_id,
            "provider": "openbmb-voxcpm2-modal",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @modal.method()
    def synthesize(self, *, voice_id: str, text: str) -> dict:
        import soundfile as sf

        clean_text = text.strip()
        if not clean_text or len(clean_text) > 1000:
            raise ValueError("Speech text must be between 1 and 1000 characters.")
        reference_id = _safe_reference_id(voice_id)
        reference_path = Path(REFERENCE_CACHE_PATH) / f"{reference_id}.wav"
        if not reference_path.is_file():
            reference_volume.reload()
        if not reference_path.is_file():
            raise ValueError("The private VoxCPM2 voice reference is unavailable.")

        wav = self.model.generate(
            text=clean_text,
            reference_wav_path=str(reference_path),
            cfg_value=2.0,
            inference_timesteps=10,
        )
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, wav, self.model.tts_model.sample_rate, format="WAV")
        return {
            "audio_bytes": wav_buffer.getvalue(),
            "provider": "openbmb-voxcpm2-modal",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @modal.method()
    def delete_reference(self, *, voice_id: str) -> None:
        reference_id = _safe_reference_id(voice_id)
        (Path(REFERENCE_CACHE_PATH) / f"{reference_id}.wav").unlink(missing_ok=True)
        reference_volume.commit()
