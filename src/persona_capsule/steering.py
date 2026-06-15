"""Request-scoped steering contracts, cache keys, and hook lifecycle helpers."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from time import monotonic
from typing import Any, Protocol

from persona_capsule.profile import ExemplarPair

MODEL_ID = "openbmb/MiniCPM4.1-8B"
MODEL_REVISION = "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3"
TRANSFORMERS_VERSION = "4.57.3"
STEERING_FORMAT_VERSION = "persona-steering-v5"
DEFAULT_LAYER_INDICES = (8, 12, 16, 20, 24)
MIN_STRENGTH = -1.5
MAX_STRENGTH = 1.5
QUALITY_WARNING_THRESHOLD = 1.1


class SteeringError(RuntimeError):
    """Base steering failure."""


class SteeringCompatibilityError(SteeringError):
    """Raised when a model cannot safely use a capsule recipe."""


class HookHandle(Protocol):
    def remove(self) -> None: ...


class HookableLayer(Protocol):
    def register_forward_hook(self, hook: Callable[..., Any]) -> HookHandle: ...


@dataclass(frozen=True, slots=True)
class SteeringRecipe:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    transformers_version: str = TRANSFORMERS_VERSION
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    layer_indices: tuple[int, ...] = DEFAULT_LAYER_INDICES
    aggregation: str = "assistant_content_mean_difference_capped_all_tokens"
    format_version: str = STEERING_FORMAT_VERSION

    def compatibility_payload(self) -> str:
        return "|".join(
            (
                self.model_id,
                self.model_revision,
                self.transformers_version,
                str(self.hidden_size),
                str(self.num_hidden_layers),
                ",".join(map(str, self.layer_indices)),
                self.aggregation,
                self.format_version,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "transformers_version": self.transformers_version,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "layer_indices": list(self.layer_indices),
            "aggregation": self.aggregation,
            "format_version": self.format_version,
        }


@dataclass(frozen=True, slots=True)
class LayerVectorDiagnostics:
    layer_index: int
    pre_normalization_norm: float
    post_normalization_norm: float
    calibration_norm: float
    component_preview: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "pre_normalization_norm": self.pre_normalization_norm,
            "post_normalization_norm": self.post_normalization_norm,
            "calibration_norm": self.calibration_norm,
            "component_preview": list(self.component_preview),
        }


@dataclass(frozen=True, slots=True)
class SteeringDiagnostics:
    recipe: SteeringRecipe
    exemplar_count: int
    derivation_hash: str
    cache_hit: bool
    strength: float
    quality_warning: str | None
    layers: tuple[LayerVectorDiagnostics, ...]
    hooks_active_after_request: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.as_dict(),
            "exemplar_count": self.exemplar_count,
            "derivation_hash": self.derivation_hash,
            "cache_hit": self.cache_hit,
            "strength": self.strength,
            "quality_warning": self.quality_warning,
            "layers": [layer.as_dict() for layer in self.layers],
            "hooks_active_after_request": self.hooks_active_after_request,
            "persistence": "request-scoped tensors; no tensor written to storage",
        }


@dataclass(frozen=True, slots=True)
class SteeringComparison:
    baseline: str
    steered: str
    diagnostics: SteeringDiagnostics


def validate_strength(strength: float) -> float:
    numeric = float(strength)
    if not MIN_STRENGTH <= numeric <= MAX_STRENGTH:
        raise SteeringError(f"Steering strength must be between {MIN_STRENGTH} and {MAX_STRENGTH}.")
    return numeric


def quality_warning(strength: float) -> str | None:
    if abs(strength) > QUALITY_WARNING_THRESHOLD:
        return (
            "Extreme steering can reduce coherence or over-amplify surface style. "
            "Compare carefully with the baseline."
        )
    return None


def validate_recipe(
    recipe: SteeringRecipe,
    *,
    model_id: str,
    model_revision: str,
    hidden_size: int,
    num_hidden_layers: int,
) -> None:
    mismatches = []
    if model_id != recipe.model_id:
        mismatches.append(f"model {model_id!r} != {recipe.model_id!r}")
    if model_revision != recipe.model_revision:
        mismatches.append(f"revision {model_revision!r} != {recipe.model_revision!r}")
    if hidden_size != recipe.hidden_size:
        mismatches.append(f"hidden size {hidden_size} != {recipe.hidden_size}")
    if num_hidden_layers != recipe.num_hidden_layers:
        mismatches.append(f"decoder layers {num_hidden_layers} != {recipe.num_hidden_layers}")
    invalid_layers = [
        index for index in recipe.layer_indices if index < 0 or index >= num_hidden_layers
    ]
    if invalid_layers:
        mismatches.append(f"invalid layer indices {invalid_layers}")
    if mismatches:
        raise SteeringCompatibilityError("; ".join(mismatches))


def normalize_components(values: Sequence[float]) -> tuple[tuple[float, ...], float]:
    components = tuple(float(value) for value in values)
    norm = sqrt(sum(value * value for value in components))
    if norm <= 1e-12:
        raise SteeringError("Cannot normalize a zero-magnitude steering direction.")
    return tuple(value / norm for value in components), norm


def compose_layer_vectors(
    first: Mapping[int, Sequence[float]],
    second: Mapping[int, Sequence[float]],
    first_weight: float,
) -> dict[int, tuple[float, ...]]:
    """Normalize each source, apply display weights, and normalize the result."""

    weight = float(first_weight)
    if not 0.0 <= weight <= 1.0:
        raise SteeringError("Fusion weight must be between 0 and 1.")
    if set(first) != set(second):
        raise SteeringCompatibilityError("Fusion sources use different steering layers.")

    composed: dict[int, tuple[float, ...]] = {}
    for layer_index in sorted(first):
        first_normalized, _ = normalize_components(first[layer_index])
        second_normalized, _ = normalize_components(second[layer_index])
        if len(first_normalized) != len(second_normalized):
            raise SteeringCompatibilityError(f"Layer {layer_index} has incompatible hidden sizes.")
        mixed = tuple(
            weight * left + (1.0 - weight) * right
            for left, right in zip(first_normalized, second_normalized, strict=True)
        )
        composed[layer_index], _ = normalize_components(mixed)
    return composed


def aggregate_pair_differences(
    differences: Mapping[int, Sequence[Sequence[float]]],
) -> tuple[dict[int, tuple[float, ...]], tuple[LayerVectorDiagnostics, ...]]:
    normalized: dict[int, tuple[float, ...]] = {}
    diagnostics: list[LayerVectorDiagnostics] = []
    for layer_index, layer_differences in sorted(differences.items()):
        if not layer_differences:
            raise SteeringError(f"Layer {layer_index} has no exemplar differences.")
        width = len(layer_differences[0])
        if width == 0 or any(len(vector) != width for vector in layer_differences):
            raise SteeringError(f"Layer {layer_index} has incompatible vector widths.")
        mean = tuple(
            sum(vector[position] for vector in layer_differences) / len(layer_differences)
            for position in range(width)
        )
        direction, pre_norm = normalize_components(mean)
        post_norm = sqrt(sum(value * value for value in direction))
        normalized[layer_index] = direction
        diagnostics.append(
            LayerVectorDiagnostics(
                layer_index=layer_index,
                pre_normalization_norm=round(pre_norm, 6),
                post_normalization_norm=round(post_norm, 6),
                calibration_norm=round(pre_norm, 6),
                component_preview=tuple(round(value, 6) for value in direction[:8]),
            )
        )
    return normalized, tuple(diagnostics)


def build_derivation_hash(
    owner_id: str,
    capsule_id: str,
    capsule_version: str,
    pairs: Sequence[ExemplarPair],
    recipe: SteeringRecipe,
) -> str:
    payload = "\n".join(
        (
            owner_id,
            capsule_id,
            capsule_version,
            recipe.compatibility_payload(),
            *(pair.pair_hash for pair in pairs),
        )
    )
    return sha256(payload.encode()).hexdigest()


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float


class ExpiringVectorCache:
    """Owner-safe warm-process cache; values never leave process memory."""

    def __init__(
        self,
        ttl_seconds: float = 300,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            self._entries.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._entries[key] = _CacheEntry(
            value=value,
            expires_at=self._clock() + self._ttl_seconds,
        )

    def invalidate_prefix(self, owner_id: str, capsule_id: str) -> None:
        prefix = sha256(f"{owner_id}\n{capsule_id}".encode()).hexdigest()[:16]
        for key in tuple(self._entries):
            if key.startswith(prefix):
                self._entries.pop(key, None)

    @staticmethod
    def namespaced_key(owner_id: str, capsule_id: str, derivation_hash: str) -> str:
        prefix = sha256(f"{owner_id}\n{capsule_id}".encode()).hexdigest()[:16]
        return f"{prefix}:{derivation_hash}"

    def __len__(self) -> int:
        return len(self._entries)


@contextmanager
def request_hook_scope(
    layers: Mapping[int, HookableLayer],
    payloads: Mapping[int, Any],
    strength: float,
    hook_factory: Callable[[Any, float], Callable[..., Any]],
):
    """Install hooks for one request and guarantee removal on every exit path."""

    validated_strength = validate_strength(strength)
    handles: list[HookHandle] = []
    try:
        if validated_strength != 0:
            for layer_index, payload in payloads.items():
                try:
                    layer = layers[layer_index]
                except KeyError as exc:
                    raise SteeringCompatibilityError(
                        f"Decoder layer {layer_index} is unavailable."
                    ) from exc
                handles.append(
                    layer.register_forward_hook(hook_factory(payload, validated_strength))
                )
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
