"""Communication-style profile models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


OCEAN_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


@dataclass
class StyleTrait:
    name: str
    score: float
    label: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleTrait:
        return cls(**data)


@dataclass
class StyleProfile:
    summary: str
    tone: str
    vocabulary: str
    cadence: str
    traits: list[StyleTrait]
    signature_phrases: list[str] = field(default_factory=list)
    palette: str = "warm amber and deep teal"
    visual_energy: str = "balanced"
    visual_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "tone": self.tone,
            "vocabulary": self.vocabulary,
            "cadence": self.cadence,
            "traits": [t.to_dict() for t in self.traits],
            "signature_phrases": self.signature_phrases,
            "palette": self.palette,
            "visual_energy": self.visual_energy,
            "visual_symbols": self.visual_symbols,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleProfile:
        return cls(
            summary=data["summary"],
            tone=data["tone"],
            vocabulary=data["vocabulary"],
            cadence=data["cadence"],
            traits=[StyleTrait.from_dict(t) for t in data.get("traits", [])],
            signature_phrases=list(data.get("signature_phrases", [])),
            palette=data.get("palette", "warm amber and deep teal"),
            visual_energy=data.get("visual_energy", "balanced"),
            visual_symbols=list(data.get("visual_symbols", [])),
        )

    def merge_with(self, other: StyleProfile, weight: float) -> StyleProfile:
        w = max(0.0, min(1.0, weight))
        trait_map = {t.name: t for t in self.traits}
        merged_traits: list[StyleTrait] = []
        for name in OCEAN_TRAITS:
            a = trait_map.get(name)
            b = next((t for t in other.traits if t.name == name), None)
            if a and b:
                score = (1 - w) * a.score + w * b.score
                merged_traits.append(
                    StyleTrait(
                        name=name,
                        score=round(score, 3),
                        label=a.label if score >= 0.5 else b.label,
                        evidence=(a.evidence[:2] + b.evidence[:2])[:3],
                    )
                )
            elif a:
                merged_traits.append(a)
            elif b:
                merged_traits.append(b)
        phrases = list(dict.fromkeys(self.signature_phrases + other.signature_phrases))[:6]
        return StyleProfile(
            summary=f"Fusion blend ({int((1-w)*100)}/{int(w*100)}): {self.summary[:120]}",
            tone=self.tone if w < 0.5 else other.tone,
            vocabulary=" | ".join(filter(None, {self.vocabulary, other.vocabulary})),
            cadence=self.cadence if w < 0.5 else other.cadence,
            traits=merged_traits,
            signature_phrases=phrases,
            palette=self.palette if w < 0.5 else other.palette,
            visual_energy=self.visual_energy if w < 0.5 else other.visual_energy,
            visual_symbols=list(dict.fromkeys(self.visual_symbols + other.visual_symbols))[:5],
        )
