"""Text generation with baseline and steered modes."""

from __future__ import annotations

import random
import re

from persona_capsule.models.capsule import CapsuleRecord, ExemplarPair
from persona_capsule.models.profile import StyleProfile
from persona_capsule.services.steering import RequestVector, derive_request_vector


class GenerationService:
    def __init__(self, use_mock: bool = True) -> None:
        self.use_mock = use_mock

    def generate_pair(
        self,
        capsule: CapsuleRecord,
        prompt: str,
        strength: float = 0.65,
    ) -> tuple[str, str]:
        vector = derive_request_vector(capsule.exemplars, capsule.steering_recipe)
        baseline = self._baseline(prompt, capsule.profile)
        steered = self._steered(prompt, capsule.profile, capsule.exemplars, vector, strength)
        return baseline, steered

    def _baseline(self, prompt: str, profile: StyleProfile) -> str:
        if self.use_mock:
            return (
                f"Baseline response to '{prompt}': "
                f"A neutral answer in plain language without strong stylistic markers."
            )
        raise NotImplementedError("Production MiniCPM runtime is not configured in this environment.")

    def _steered(
        self,
        prompt: str,
        profile: StyleProfile,
        exemplars: list[ExemplarPair],
        vector: RequestVector,
        strength: float,
    ) -> str:
        if self.use_mock:
            phrase = random.choice(profile.signature_phrases or ["honestly"])
            sample = exemplars[0].style_example if exemplars else profile.summary
            cadence_hint = "..." if profile.cadence.startswith("short") else " — "
            return (
                f"Steered ({strength:.2f}, cache={vector.cache_key}) response to '{prompt}': "
                f"{profile.tone.capitalize()}{cadence_hint} leaning on '{phrase}'. "
                f"Echoes approved style like: \"{sample[:100]}\""
            )
        raise NotImplementedError("Production MiniCPM runtime is not configured in this environment.")

    @staticmethod
    def apply_style_to_text(text: str, profile: StyleProfile, strength: float) -> str:
        styled = text
        if strength >= 0.5 and profile.signature_phrases:
            styled = f"{profile.signature_phrases[0].capitalize()} — {styled}"
        if profile.tone.startswith("enthusiastic"):
            styled = re.sub(r"\.$", "!", styled)
        if profile.vocabulary.startswith("casual"):
            styled = styled.replace("do not", "don't").replace("cannot", "can't")
        return styled
