"""Capsule fusion."""

from __future__ import annotations

from persona_capsule.models.capsule import CapsuleRecord, CreationMode, ExemplarPair
from persona_capsule.services.steering import compose_vectors, derive_request_vector


def fuse_capsules(
    left: CapsuleRecord,
    right: CapsuleRecord,
    owner_id: str,
    display_name: str,
    right_weight: float,
) -> CapsuleRecord:
    if left.steering_recipe.model_id != right.steering_recipe.model_id:
        raise ValueError("Capsules must share the same MiniCPM model recipe.")
    if left.steering_recipe.layer_indices != right.steering_recipe.layer_indices:
        raise ValueError("Capsules must share the same steering layer configuration.")

    left_vector = derive_request_vector(left.exemplars, left.steering_recipe)
    right_vector = derive_request_vector(right.exemplars, right.steering_recipe)
    compose_vectors(left_vector, right_vector, 1.0 - right_weight)

    merged_profile = left.profile.merge_with(right.profile, right_weight)
    merged_exemplars = (
        left.exemplars[:2] + right.exemplars[:2]
    )[:4] or [ExemplarPair(style_example=left.profile.summary, neutral_contrast=right.profile.summary)]

    fused = CapsuleRecord.new(
        owner_id=owner_id,
        display_name=display_name,
        profile=merged_profile,
        exemplars=merged_exemplars,
        creation_mode=CreationMode.FUSION,
    )
    fused.provenance = [left.id, right.id]
    fused.fusion_weights = {left.id: round(1.0 - right_weight, 2), right.id: round(right_weight, 2)}
    fused.steering_recipe = left.steering_recipe
    return fused
