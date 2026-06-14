"""Asynchronous Modal QLoRA training for optional Deep Capsules."""

import difflib
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import modal

APP_NAME = "persona-capsule-deep"
MODEL_ID = "openbmb/MiniCPM4.1-8B"
MODEL_REVISION = "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3"
MODEL_CACHE_PATH = "/models"
OUTPUT_ROOT = "/outputs"

model_volume = modal.Volume.from_name("persona-capsule-hf-models", create_if_missing=True)
output_volume = modal.Volume.from_name("persona-capsule-deep-outputs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.10.1",
        "bitsandbytes==0.49.0",
        "datasets==4.0.0",
        "huggingface-hub[hf-xet]==0.36.0",
        "peft==0.18.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "torch==2.7.1",
        "transformers==4.57.3",
        "trl==0.29.0",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)
app = modal.App(APP_NAME)


def _repair_neutral_contrast(positive: str, neutral: str) -> str:
    clean_positive = " ".join(positive.split())
    clean_neutral = " ".join(neutral.split())
    if clean_positive.casefold() != clean_neutral.casefold():
        return clean_neutral
    return f"State this information plainly: {clean_positive}"


def _find_content_span(sequence: list[int], content: list[int]) -> tuple[int, int] | None:
    if not content or len(content) > len(sequence):
        return None
    for start in range(len(sequence) - len(content), -1, -1):
        if sequence[start : start + len(content)] == content:
            return start, start + len(content)
    return None


def _memorization(output: str, references: list[str]) -> float:
    return max(
        (difflib.SequenceMatcher(None, output, reference).ratio() for reference in references),
        default=0.0,
    )


def _style_score(output: str, references: list[str]) -> float:
    """Compare broad writing shape without rewarding copied phrases."""

    def features(text: str) -> tuple[float, ...]:
        words = text.split()
        sentences = [part for part in text.replace("!", ".").replace("?", ".").split(".") if part]
        word_count = max(1, len(words))
        sentence_count = max(1, len(sentences))
        return (
            min(word_count / sentence_count, 40) / 40,
            min(text.count("!") / sentence_count, 2) / 2,
            min(text.count("?") / sentence_count, 2) / 2,
            min(text.count(",") / sentence_count, 4) / 4,
            sum(word.isupper() for word in words) / word_count,
        )

    if not references:
        return 0.0
    target_rows = [features(reference) for reference in references]
    target = tuple(sum(row[index] for row in target_rows) / len(target_rows) for index in range(5))
    observed = features(output)
    distance = sum(abs(left - right) for left, right in zip(observed, target, strict=True))
    return max(0.0, 1.0 - distance / len(target))


@app.function(
    image=image,
    gpu="A10G",
    timeout=90 * 60,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
    volumes={MODEL_CACHE_PATH: model_volume, OUTPUT_ROOT: output_volume},
    secrets=[modal.Secret.from_name("persona-capsule-huggingface")],
)
def train_deep_capsule(payload: dict) -> dict:
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    pairs = list(payload.get("training_pairs", ()))
    if len(pairs) < 2:
        raise ValueError("Deep Capsule requires at least two approved pairs.")
    key = str(payload["idempotency_key"])
    capsule_id = str(payload["capsule_id"])
    output_dir = Path(OUTPUT_ROOT) / hashlib.sha256(key.encode()).hexdigest()[:24]
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": "cuda"},
    )
    rows = [
        {
            "prompt": [{"role": "user", "content": pair["neutral"]}],
            "completion": [{"role": "assistant", "content": pair["positive"]}],
        }
        for pair in pairs
    ]
    split = max(1, len(rows) - 1)
    train_dataset = Dataset.from_list(rows[:split])
    eval_dataset = Dataset.from_list(rows[split:])
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    config = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max(12, len(train_dataset) * 8),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=2,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        max_length=512,
        seed=7613,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir, safe_serialization=True)

    held_out = pairs[-1]
    layer_indices = tuple(
        index
        for index in (8, 12, 16, 20, 24)
        if index < int(trainer.model.config.num_hidden_layers)
    )
    base_model = trainer.model.get_base_model()
    layers = {index: base_model.model.layers[index] for index in layer_indices}

    def activation(text: str) -> dict[int, Any]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
        with trainer.model.disable_adapter(), torch.inference_mode():
            outputs = trainer.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        content_span = _find_content_span(input_ids, content_ids)
        if content_span is None:
            non_padding = inputs["attention_mask"][0].nonzero().flatten()
            end = max(1, int(non_padding[-1].item()))
            start = max(0, end - max(1, len(content_ids)))
        else:
            start, end = content_span
        return {
            index: outputs.hidden_states[index + 1][0, start:end].float().mean(dim=0)
            for index in layer_indices
        }

    steering_pairs = pairs[:split]
    differences = {index: [] for index in layer_indices}
    for pair in steering_pairs:
        positive = activation(str(pair["positive"]))
        neutral = activation(
            _repair_neutral_contrast(
                str(pair["positive"]),
                str(pair["neutral"]),
            )
        )
        for index in layer_indices:
            differences[index].append(positive[index] - neutral[index])
    directions = {}
    for index in layer_indices:
        mean_difference = torch.stack(differences[index]).mean(dim=0)
        norm = torch.linalg.vector_norm(mean_difference)
        if float(norm.item()) <= 1e-8:
            raise RuntimeError(f"Layer {index} produced a zero-magnitude steering direction.")
        directions[index] = (mean_difference / norm).to(device="cuda", dtype=torch.bfloat16)

    def hook_factory(direction: Any, strength: float):
        def steer(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden_states = output[0] if isinstance(output, tuple) else output
            scaled = direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).view(1, 1, -1)
            steered = hidden_states + strength * scaled
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        return steer

    @contextmanager
    def steering_scope(strength: float = 0.8) -> Iterator[None]:
        handles = [
            layers[index].register_forward_hook(hook_factory(direction, strength))
            for index, direction in directions.items()
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate(*, adapter_enabled: bool, steering_enabled: bool) -> str:
        messages = [{"role": "user", "content": held_out["neutral"]}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
        adapter_scope = nullcontext() if adapter_enabled else trainer.model.disable_adapter()
        live_steering_scope = steering_scope() if steering_enabled else nullcontext()
        with adapter_scope, live_steering_scope, torch.inference_mode():
            generated = trainer.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=96,
            )
        return tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

    base = generate(adapter_enabled=False, steering_enabled=False)
    steering_only = generate(adapter_enabled=False, steering_enabled=True)
    lora_only = generate(adapter_enabled=True, steering_enabled=False)
    combined = generate(adapter_enabled=True, steering_enabled=True)
    held_out_references = [str(held_out["positive"])]
    training_references = [str(pair["positive"]) for pair in steering_pairs]
    base_style_score = _style_score(base, held_out_references)
    combined_style_score = _style_score(combined, held_out_references)
    evaluation = {
        "quality_gain": round(combined_style_score - base_style_score, 4),
        "memorization_score": round(_memorization(combined, training_references), 4),
        "held_out_count": 1,
        "activation_layers": list(layer_indices),
        "steering_request_scoped": True,
        "outputs": {
            "base": base,
            "live_steering": steering_only,
            "lora_only": lora_only,
            "combined": combined,
        },
    }

    api = HfApi(token=os.environ["HF_TOKEN"])
    owner = str(api.whoami()["name"])
    prefix = os.environ.get("HF_DEEP_REPO_PREFIX", "persona-capsule")
    repo_id = f"{owner}/{prefix}-{capsule_id[:12]}-minicpm-lora"
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message=f"Deep Capsule adapter {capsule_id}",
    )
    output_volume.commit()
    return {
        "status": "completed",
        "progress": 100,
        "message": "Training and held-out evaluation completed.",
        "evaluation": evaluation,
        "artifact": {
            "repo_id": repo_id,
            "revision": commit.oid,
            "format": "peft-safetensors",
            "base_model_id": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
        },
        "visual_lora": {
            "requested": bool(payload.get("visual_lora")),
            "status": "needs-approved-image-dataset"
            if payload.get("visual_lora")
            else "not-requested",
            "message": (
                "No visual adapter was attached. Personal FLUX training requires a separate "
                "reviewed image dataset; the global FLUX card path remains available."
            ),
        },
    }
