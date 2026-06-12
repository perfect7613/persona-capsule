"""Deterministic demo capsule used before model-backed creation is available."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class DemoCapsule:
    name: str
    archetype: str
    signal: str
    traits: tuple[str, ...]
    cadence: str
    steering_recipe: str


DEMO_CAPSULE = DemoCapsule(
    name="Signal / No. 01",
    archetype="The Clear-Eyed Builder",
    signal="Warm precision with a bias toward useful action.",
    traits=("direct", "curious", "grounded", "quietly playful"),
    cadence="Short setup. Concrete detail. A clean landing.",
    steering_recipe="Live contrast pair · request-scoped · no stored activations",
)


def demo_reply(prompt: str, intensity: str) -> str:
    """Return a stable, capsule-styled response without calling a model."""

    clean_prompt = " ".join(prompt.split()).strip()
    if not clean_prompt:
        return "Give the capsule a situation first. A little context makes the voice legible."

    fingerprint = sha256(f"{clean_prompt}:{intensity}".encode()).hexdigest()[:8].upper()
    preface = {
        "Subtle": "Let’s make this easy to act on.",
        "Balanced": "Here’s the shape of it, without the fog.",
        "Expressive": "Good, there’s a real signal in this. Let’s sharpen it.",
    }[intensity]
    return (
        f"**{preface}**\n\n"
        f"You asked: “{clean_prompt}”\n\n"
        "The working move is to name the outcome, remove one unnecessary choice, "
        "and finish with the next concrete action. This response is deterministic; "
        "live MiniCPM steering arrives in Slice 4.\n\n"
        f"`demo trace {fingerprint}`"
    )
