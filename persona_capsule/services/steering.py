"""Request-scoped steering vector derivation and composition."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from persona_capsule.models.capsule import ExemplarPair, SteeringRecipe


@dataclass
class RequestVector:
    layer_vectors: dict[int, list[float]]
    recipe: SteeringRecipe
    cache_key: str


def _hash_exemplars(exemplars: list[ExemplarPair], recipe: SteeringRecipe) -> str:
    payload = "|".join(f"{e.style_example}::{e.neutral_contrast}" for e in exemplars)
    payload += recipe.to_dict().__repr__()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _text_signature(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for idx, ch in enumerate(text.encode("utf-8")):
        vec[idx % dim] += (ch % 17) / 17.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def derive_request_vector(
    exemplars: list[ExemplarPair],
    recipe: SteeringRecipe,
) -> RequestVector:
    if not exemplars:
        raise ValueError("Approved exemplars are required for steering derivation.")

    pos = _text_signature(" ".join(e.style_example for e in exemplars))
    neg = _text_signature(" ".join(e.neutral_contrast for e in exemplars))
    direction = [p - n for p, n in zip(pos, neg)]
    norm = math.sqrt(sum(v * v for v in direction)) or 1.0
    direction = [v / norm for v in direction]

    layer_vectors: dict[int, list[float]] = {}
    for layer in recipe.layer_indices:
        scale = 1.0 + (layer / max(recipe.layer_indices[-1], 1)) * 0.15
        layer_vectors[layer] = [v * scale for v in direction]

    return RequestVector(
        layer_vectors=layer_vectors,
        recipe=recipe,
        cache_key=_hash_exemplars(exemplars, recipe),
    )


def compose_vectors(
    left: RequestVector,
    right: RequestVector,
    left_weight: float,
) -> RequestVector:
    if left.recipe.model_id != right.recipe.model_id:
        raise ValueError("Fusion requires identical MiniCPM model revisions.")
    if left.recipe.layer_indices != right.recipe.layer_indices:
        raise ValueError("Fusion requires identical steering layer sets.")

    w = max(0.0, min(1.0, left_weight))
    combined: dict[int, list[float]] = {}
    for layer in left.layer_vectors:
        lv = left.layer_vectors[layer]
        rv = right.layer_vectors[layer]
        merged = [(1 - w) * a + w * b for a, b in zip(lv, rv)]
        norm = math.sqrt(sum(v * v for v in merged)) or 1.0
        combined[layer] = [v / norm for v in merged]

    return RequestVector(
        layer_vectors=combined,
        recipe=left.recipe,
        cache_key=f"{left.cache_key}:{right.cache_key}:{w:.2f}",
    )
