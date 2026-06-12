"""Modal GPU runtime for request-scoped MiniCPM4.1 activation steering."""

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).parent / "src"))

from persona_capsule.profile import ExemplarPair
from persona_capsule.steering import (
    ExpiringVectorCache,
    LayerVectorDiagnostics,
    SteeringDiagnostics,
    SteeringRecipe,
    build_derivation_hash,
    quality_warning,
    request_hook_scope,
    validate_recipe,
    validate_strength,
)

APP_NAME = "persona-capsule-minicpm"
MODEL_CACHE_PATH = "/models"
MODEL_VOLUME = modal.Volume.from_name(
    "persona-capsule-hf-models",
    create_if_missing=True,
)
RECIPE = SteeringRecipe()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.10.1",
        "huggingface-hub[hf-xet]==0.36.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "torch==2.7.1",
        f"transformers=={RECIPE.transformers_version}",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("persona_capsule")
)

hf_token = os.environ.get("HF_TOKEN")
secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token})] if hf_token else []
app = modal.App(APP_NAME)


def _decode_new_tokens(tokenizer: Any, input_ids: Any, generated: Any) -> str:
    new_tokens = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.cls(
    image=image,
    gpu="A10G",
    timeout=30 * 60,
    scaledown_window=5 * 60,
    volumes={MODEL_CACHE_PATH: MODEL_VOLUME},
    secrets=secrets,
)
class MiniCPMSteeringRuntime:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(
            RECIPE.model_id,
            revision=RECIPE.model_revision,
            trust_remote_code=True,
        )
        # Short hackathon prompts do not require the optional sparse-attention kernels.
        config.sparse_config = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            RECIPE.model_id,
            revision=RECIPE.model_revision,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            RECIPE.model_id,
            revision=RECIPE.model_revision,
            config=config,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
        ).eval()
        validate_recipe(
            RECIPE,
            model_id=RECIPE.model_id,
            model_revision=RECIPE.model_revision,
            hidden_size=self.model.config.hidden_size,
            num_hidden_layers=self.model.config.num_hidden_layers,
        )
        self.layers = {index: self.model.model.layers[index] for index in RECIPE.layer_indices}
        self.cache = ExpiringVectorCache(ttl_seconds=300)

    def _tokenize_chat(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        return self.tokenizer(rendered, return_tensors="pt").to("cuda")

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self._tokenize_chat(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            repetition_penalty=1.05,
        )
        return _decode_new_tokens(self.tokenizer, inputs["input_ids"], generated)

    def _activation(self, text: str) -> dict[int, Any]:
        import torch

        inputs = self._tokenize_chat(
            [{"role": "assistant", "content": text}],
            add_generation_prompt=False,
        )
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        last_position = int(inputs["attention_mask"][0].nonzero()[-1].item())
        return {
            index: outputs.hidden_states[index + 1][0, last_position].float()
            for index in RECIPE.layer_indices
        }

    def _derive(
        self,
        owner_id: str,
        capsule_id: str,
        capsule_version: str,
        pairs: Sequence[ExemplarPair],
    ) -> tuple[dict[int, Any], tuple[LayerVectorDiagnostics, ...], str, bool]:
        import torch

        derivation_hash = build_derivation_hash(
            owner_id,
            capsule_id,
            capsule_version,
            pairs,
            RECIPE,
        )
        cache_key = self.cache.namespaced_key(
            owner_id,
            capsule_id,
            derivation_hash,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            vectors, diagnostics = cached
            return vectors, diagnostics, derivation_hash, True

        differences = {index: [] for index in RECIPE.layer_indices}
        for pair in pairs:
            positive = self._activation(pair.positive)
            neutral = self._activation(pair.neutral)
            for index in RECIPE.layer_indices:
                differences[index].append(positive[index] - neutral[index])
            del positive, neutral

        vectors: dict[int, Any] = {}
        diagnostics = []
        for index in RECIPE.layer_indices:
            mean_difference = torch.stack(differences[index]).mean(dim=0)
            pre_norm = float(torch.linalg.vector_norm(mean_difference).item())
            if pre_norm <= 1e-8:
                raise RuntimeError(f"Layer {index} produced a zero-magnitude steering direction.")
            direction = mean_difference / pre_norm
            post_norm = float(torch.linalg.vector_norm(direction).item())
            vectors[index] = direction.to(dtype=torch.bfloat16, device="cuda")
            diagnostics.append(
                LayerVectorDiagnostics(
                    layer_index=index,
                    pre_normalization_norm=round(pre_norm, 6),
                    post_normalization_norm=round(post_norm, 6),
                    component_preview=tuple(
                        round(float(value), 6) for value in direction[:8].detach().cpu().tolist()
                    ),
                )
            )
        diagnostics_tuple = tuple(diagnostics)
        self.cache.set(cache_key, (vectors, diagnostics_tuple))
        return vectors, diagnostics_tuple, derivation_hash, False

    @staticmethod
    def _hook_factory(direction: Any, strength: float):
        def steer(_module: Any, _inputs: Any, output: tuple[Any, ...]):
            hidden_states = output[0]
            scaled = direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).view(1, 1, -1)
            return (hidden_states + strength * scaled, *output[1:])

        return steer

    @modal.method()
    def compare(
        self,
        *,
        owner_id: str,
        capsule_id: str,
        capsule_version: str,
        prompt: str,
        pairs: list[dict[str, Any]],
        strength: float = 0.8,
        max_new_tokens: int = 96,
    ) -> dict[str, Any]:
        started_at = monotonic()
        validated_strength = validate_strength(strength)
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not 1 <= len(pairs) <= 6:
            raise ValueError("Provide between one and six approved exemplar pairs.")
        exemplar_pairs = tuple(
            ExemplarPair(
                positive=str(pair["positive"]).strip(),
                neutral=str(pair["neutral"]).strip(),
                source_index=int(pair.get("source_index", index)),
            )
            for index, pair in enumerate(pairs)
        )
        if any(not pair.positive or not pair.neutral for pair in exemplar_pairs):
            raise ValueError("Every exemplar pair requires positive and neutral text.")

        baseline = self._generate(prompt.strip(), max_new_tokens)
        vectors, layers, derivation_hash, cache_hit = self._derive(
            owner_id,
            capsule_id,
            capsule_version,
            exemplar_pairs,
        )
        hooks_active_after_request = True
        try:
            with request_hook_scope(
                self.layers,
                vectors,
                validated_strength,
                self._hook_factory,
            ):
                steered = self._generate(prompt.strip(), max_new_tokens)
        finally:
            hooks_active_after_request = any(
                bool(getattr(layer, "_forward_hooks", {})) for layer in self.layers.values()
            )

        diagnostics = SteeringDiagnostics(
            recipe=RECIPE,
            exemplar_count=len(exemplar_pairs),
            derivation_hash=derivation_hash,
            cache_hit=cache_hit,
            strength=validated_strength,
            quality_warning=quality_warning(validated_strength),
            layers=layers,
            hooks_active_after_request=hooks_active_after_request,
        )
        return {
            "baseline": baseline,
            "steered": steered,
            "diagnostics": diagnostics.as_dict(),
            "runtime": {
                "gpu": "A10G",
                "dense_short_context_mode": True,
                "elapsed_seconds": round(monotonic() - started_at, 3),
            },
        }

    @modal.method()
    def invalidate(self, *, owner_id: str, capsule_id: str) -> None:
        self.cache.invalidate_prefix(owner_id, capsule_id)


@app.local_entrypoint()
def main(
    prompt: str = "Explain why a small team should test the risky assumption first.",
    strength: float = 0.85,
) -> None:
    sample_pairs = [
        {
            "positive": (
                "Good, there is a real signal here. Let us name the risky assumption, "
                "test the smallest useful version, and make the next move obvious."
            ),
            "neutral": (
                "Identify the risky assumption, test a small version, and define the next step."
            ),
            "source_index": 0,
        },
        {
            "positive": (
                "I am not convinced yet; what evidence would change our minds? "
                "That answer should drive the experiment."
            ),
            "neutral": (
                "Determine what evidence would change the decision and use it "
                "to design the experiment."
            ),
            "source_index": 1,
        },
        {
            "positive": (
                "Short setup. Concrete detail. A clean landing. "
                "We can keep the thinking rigorous without making it heavy."
            ),
            "neutral": ("Use a concise introduction, specific details, and a clear conclusion."),
            "source_index": 2,
        },
    ]
    result = MiniCPMSteeringRuntime().compare.remote(
        owner_id="hf:steering-spike",
        capsule_id="signal-01",
        capsule_version="synthetic-v1",
        prompt=prompt,
        pairs=sample_pairs,
        strength=strength,
        max_new_tokens=96,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
