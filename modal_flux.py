"""Modal FLUX.2 Klein runtime for generated capsule artwork."""

import io
import os

import modal

APP_NAME = "persona-capsule-flux"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
ANIME_LORA_ID = "Sawata97/flux2_4b_koni_animestyle"
ANIME_LORA_REVISION = "b392813770c6155d3ae22d3fb8062f247f5108a2"
ANIME_LORA_WEIGHT = "Flux_klein_4b_anime_Koni.safetensors"
ANIME_LORA_SCALE = 0.8
MODEL_CACHE_PATH = "/models"

model_volume = modal.Volume.from_name(
    "persona-capsule-flux-models",
    create_if_missing=True,
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.10.1",
        "diffusers==0.37.1",
        "huggingface-hub[hf-xet]==0.36.0",
        "peft==0.17.1",
        "pillow==12.0.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "torch==2.7.1",
        "transformers==4.57.3",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)
hf_token = os.environ.get("HF_TOKEN")
secret_payload = {"HF_TOKEN": hf_token} if hf_token else {}
secrets = [modal.Secret.from_dict(secret_payload)] if secret_payload else []
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=20 * 60,
    scaledown_window=5 * 60,
    volumes={MODEL_CACHE_PATH: model_volume},
    secrets=secrets,
)
class FluxCardRuntime:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")
        self.pipe.load_lora_weights(
            ANIME_LORA_ID,
            revision=ANIME_LORA_REVISION,
            weight_name=ANIME_LORA_WEIGHT,
            adapter_name="anime_persona",
        )
        self.pipe.set_adapters(
            "anime_persona",
            adapter_weights=ANIME_LORA_SCALE,
        )

    @modal.method()
    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        width: int = 768,
        height: int = 768,
    ) -> dict:
        import torch

        if not prompt.strip() or len(prompt) > 1600:
            raise ValueError("Card prompt must contain between 1 and 1600 characters.")
        if width % 16 or height % 16:
            raise ValueError("Image dimensions must be multiples of 16.")
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        generated = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=generator,
        ).images[0]
        output = io.BytesIO()
        generated.save(output, format="PNG", optimize=True)
        return {
            "png_bytes": output.getvalue(),
            "provider": "flux2-klein-anime-lora-modal",
            "model_id": f"{MODEL_ID}+adapter:{ANIME_LORA_ID}",
            "model_revision": f"{MODEL_REVISION}+adapter:{ANIME_LORA_REVISION}",
            "lora_loaded": True,
            "lora_weight": ANIME_LORA_WEIGHT,
            "lora_scale": ANIME_LORA_SCALE,
        }
