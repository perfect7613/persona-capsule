import pytest

from persona_capsule.profile import ExemplarPair
from persona_capsule.steering import (
    ExpiringVectorCache,
    SteeringCompatibilityError,
    SteeringError,
    SteeringRecipe,
    aggregate_pair_differences,
    build_derivation_hash,
    quality_warning,
    request_hook_scope,
    validate_recipe,
    validate_strength,
)


class FakeHandle:
    def __init__(self, layer: "FakeLayer") -> None:
        self.layer = layer

    def remove(self) -> None:
        self.layer.active_hooks -= 1


class FakeLayer:
    def __init__(self) -> None:
        self.active_hooks = 0

    def register_forward_hook(self, hook):
        del hook
        self.active_hooks += 1
        return FakeHandle(self)


def test_derivation_averages_and_normalizes_each_layer() -> None:
    vectors, diagnostics = aggregate_pair_differences(
        {
            8: [(3, 0, 0), (1, 0, 0)],
            12: [(0, 4, 0), (0, 2, 0)],
        }
    )

    assert vectors == {8: (1.0, 0.0, 0.0), 12: (0.0, 1.0, 0.0)}
    assert all(item.post_normalization_norm == 1 for item in diagnostics)
    assert diagnostics[0].pre_normalization_norm == 2
    assert diagnostics[1].pre_normalization_norm == 3


def test_zero_strength_installs_no_hooks() -> None:
    layers = {8: FakeLayer()}

    with request_hook_scope(layers, {8: object()}, 0, lambda payload, strength: object()):
        assert layers[8].active_hooks == 0

    assert layers[8].active_hooks == 0


def test_hooks_are_removed_after_success_and_failure() -> None:
    layers = {8: FakeLayer(), 12: FakeLayer()}

    def hook_factory(payload, strength):
        return payload, strength

    with request_hook_scope(layers, {8: "a", 12: "b"}, 0.8, hook_factory):
        assert sum(layer.active_hooks for layer in layers.values()) == 2

    assert sum(layer.active_hooks for layer in layers.values()) == 0

    with pytest.raises(RuntimeError):
        with request_hook_scope(layers, {8: "a", 12: "b"}, 0.8, hook_factory):
            raise RuntimeError("generation failed")

    assert sum(layer.active_hooks for layer in layers.values()) == 0


def test_cache_is_owner_safe_expires_and_invalidates() -> None:
    now = [100.0]
    cache = ExpiringVectorCache(ttl_seconds=10, clock=lambda: now[0])
    alice_key = cache.namespaced_key("hf:alice", "capsule", "derivation")
    bob_key = cache.namespaced_key("hf:bob", "capsule", "derivation")

    cache.set(alice_key, "alice-vector")
    cache.set(bob_key, "bob-vector")

    assert cache.get(alice_key) == "alice-vector"
    assert cache.get(bob_key) == "bob-vector"
    cache.invalidate_prefix("hf:alice", "capsule")
    assert cache.get(alice_key) is None
    assert cache.get(bob_key) == "bob-vector"

    now[0] = 111
    assert cache.get(bob_key) is None


def test_pair_edits_change_the_derivation_hash() -> None:
    recipe = SteeringRecipe()
    first = ExemplarPair("Warm and direct!", "The response is direct.", 0)
    edited = ExemplarPair("Warm, direct, and brief!", "The response is direct.", 0)

    first_hash = build_derivation_hash("hf:owner", "capsule", "v1", [first], recipe)
    edited_hash = build_derivation_hash("hf:owner", "capsule", "v1", [edited], recipe)

    assert first_hash != edited_hash


def test_incompatible_recipe_and_strength_are_rejected() -> None:
    recipe = SteeringRecipe()

    with pytest.raises(SteeringCompatibilityError, match="hidden size"):
        validate_recipe(
            recipe,
            model_id=recipe.model_id,
            model_revision=recipe.model_revision,
            hidden_size=1024,
            num_hidden_layers=32,
        )

    with pytest.raises(SteeringError, match="between"):
        validate_strength(2.0)

    assert quality_warning(1.3) is not None
    assert quality_warning(0.7) is None
