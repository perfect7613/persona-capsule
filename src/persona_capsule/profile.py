"""Canonical style-profile and exemplar models."""

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any


def ensure_distinct_contrast(positive: str, neutral: str) -> tuple[str, bool]:
    clean_positive = " ".join(positive.split())
    clean_neutral = " ".join(neutral.split())
    if clean_positive.casefold() != clean_neutral.casefold():
        return clean_neutral, False
    return f"State this information plainly: {clean_positive}", True


@dataclass(frozen=True, slots=True)
class StyleDimensions:
    openness: float
    conscientiousness: float
    expressiveness: float
    agreeableness: float
    emotional_range: float
    directness: float
    formality: float

    def as_dict(self) -> dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "expressiveness": self.expressiveness,
            "agreeableness": self.agreeableness,
            "emotional_range": self.emotional_range,
            "directness": self.directness,
            "formality": self.formality,
        }

    def edited(self, values: dict[str, float]) -> "StyleDimensions":
        bounded = {key: max(0.0, min(100.0, float(value))) for key, value in values.items()}
        return replace(self, **bounded)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleDimensions":
        return cls(
            openness=float(payload["openness"]),
            conscientiousness=float(payload["conscientiousness"]),
            expressiveness=float(payload["expressiveness"]),
            agreeableness=float(payload["agreeableness"]),
            emotional_range=float(payload["emotional_range"]),
            directness=float(payload["directness"]),
            formality=float(payload["formality"]),
        )


@dataclass(frozen=True, slots=True)
class StyleProfile:
    summary: str
    descriptors: tuple[str, ...]
    lexical_tendencies: tuple[str, ...]
    sentence_rhythm: str
    dimensions: StyleDimensions
    evidence: tuple[str, ...]
    uncertainty: float

    def as_dict(self, *, include_private_evidence: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary": self.summary,
            "descriptors": list(self.descriptors),
            "lexical_tendencies": list(self.lexical_tendencies),
            "sentence_rhythm": self.sentence_rhythm,
            "dimensions": self.dimensions.as_dict(),
            "uncertainty": self.uncertainty,
        }
        if include_private_evidence:
            payload["evidence"] = list(self.evidence)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleProfile":
        return cls(
            summary=str(payload["summary"]),
            descriptors=tuple(map(str, payload.get("descriptors", ()))),
            lexical_tendencies=tuple(map(str, payload.get("lexical_tendencies", ()))),
            sentence_rhythm=str(payload["sentence_rhythm"]),
            dimensions=StyleDimensions.from_dict(payload["dimensions"]),
            evidence=tuple(map(str, payload.get("evidence", ()))),
            uncertainty=float(payload["uncertainty"]),
        )


@dataclass(frozen=True, slots=True)
class ExemplarPair:
    positive: str
    neutral: str
    source_index: int

    @property
    def pair_hash(self) -> str:
        content = f"{self.source_index}\0{self.positive}\0{self.neutral}"
        return sha256(content.encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive,
            "neutral": self.neutral,
            "source_index": self.source_index,
            "pair_hash": self.pair_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExemplarPair":
        return cls(
            positive=str(payload["positive"]),
            neutral=str(payload["neutral"]),
            source_index=int(payload["source_index"]),
        )


@dataclass(frozen=True, slots=True)
class ApprovedProfile:
    profile: StyleProfile
    exemplar_pairs: tuple[ExemplarPair, ...]
    source_fingerprint: str
