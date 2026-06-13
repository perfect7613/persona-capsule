"""Infer editable communication-style profiles from user messages."""

from __future__ import annotations

import re
from collections import Counter

from persona_capsule.models.capsule import ExemplarPair
from persona_capsule.models.profile import OCEAN_TRAITS, StyleProfile, StyleTrait

EXCLAMATION_WORDS = {"!", "!!", "!!!"}
CASUAL_MARKERS = {"lol", "haha", "tbh", "imo", "ngl", "kinda", "gonna", "wanna"}
FORMAL_MARKERS = {"therefore", "however", "furthermore", "regarding", "respectfully"}
WARM_MARKERS = {"thanks", "thank you", "appreciate", "love", "happy", "glad"}
ANXIOUS_MARKERS = {"sorry", "worried", "nervous", "maybe", "probably", "might"}


def _score_trait(messages: list[str], markers: set[str]) -> float:
    if not messages:
        return 0.5
    hits = 0
    total = 0
    for msg in messages:
        words = re.findall(r"[A-Za-z']+", msg.lower())
        total += max(len(words), 1)
        hits += sum(1 for w in words if w in markers)
    return min(1.0, 0.35 + hits / max(total, 1) * 4)


def _avg_sentence_length(messages: list[str]) -> float:
    lengths = [len(re.findall(r"[.!?]+", m)) or 1 for m in messages]
    words = [len(m.split()) for m in messages]
    return sum(words) / sum(lengths)


def _pick_evidence(messages: list[str], limit: int = 2) -> list[str]:
    ranked = sorted(messages, key=len, reverse=True)
    return ranked[:limit]


def infer_profile(messages: list[str]) -> StyleProfile:
    if not messages:
        raise ValueError("At least one message is required.")

    avg_len = _avg_sentence_length(messages)
    exclamations = sum(m.count("!") for m in messages)
    questions = sum(m.count("?") for m in messages)
    lowercase_ratio = sum(1 for m in messages if m == m.lower()) / len(messages)

    tone = "warm and conversational"
    if exclamations > len(messages):
        tone = "enthusiastic and expressive"
    elif questions > len(messages):
        tone = "curious and reflective"
    elif avg_len > 18:
        tone = "thoughtful and measured"

    vocabulary = "plain and direct"
    if _score_trait(messages, FORMAL_MARKERS) > 0.55:
        vocabulary = "formal and precise"
    elif _score_trait(messages, CASUAL_MARKERS) > 0.45:
        vocabulary = "casual and playful"

    cadence = "medium-length sentences"
    if avg_len <= 10:
        cadence = "short, punchy bursts"
    elif avg_len >= 20:
        cadence = "longer, flowing sentences"

    trait_scores = {
        "openness": min(1.0, 0.4 + questions / max(len(messages), 1) * 0.2),
        "conscientiousness": _score_trait(messages, FORMAL_MARKERS),
        "extraversion": min(1.0, 0.35 + exclamations / max(len(messages), 1) * 0.25),
        "agreeableness": _score_trait(messages, WARM_MARKERS),
        "neuroticism": _score_trait(messages, ANXIOUS_MARKERS),
    }
    labels = {
        "openness": ("imaginative", "practical"),
        "conscientiousness": ("structured", "flexible"),
        "extraversion": ("outgoing", "reserved"),
        "agreeableness": ("supportive", "direct"),
        "neuroticism": ("cautious", "steady"),
    }
    traits = [
        StyleTrait(
            name=name,
            score=round(trait_scores[name], 3),
            label=labels[name][0] if trait_scores[name] >= 0.5 else labels[name][1],
            evidence=_pick_evidence(messages, 1),
        )
        for name in OCEAN_TRAITS
    ]

    phrase_counts = Counter()
    for msg in messages:
        for phrase in re.findall(r"[A-Za-z']{3,}", msg.lower()):
            if phrase not in CASUAL_MARKERS and len(phrase) > 4:
                phrase_counts[phrase] += 1
    signature = [p for p, _ in phrase_counts.most_common(5)]

    summary = (
        f"A {tone} communicator with {vocabulary} vocabulary and {cadence}. "
        f"Messages skew {'informal' if lowercase_ratio > 0.5 else 'mixed register'}."
    )

    palette = "warm amber and deep teal"
    if trait_scores["openness"] > 0.6:
        palette = "violet gradients with gold accents"
    if trait_scores["conscientiousness"] > 0.65:
        palette = "slate blue with crisp white highlights"

    return StyleProfile(
        summary=summary,
        tone=tone,
        vocabulary=vocabulary,
        cadence=cadence,
        traits=traits,
        signature_phrases=signature,
        palette=palette,
        visual_energy="high" if trait_scores["extraversion"] > 0.6 else "balanced",
        visual_symbols=["chat bubble", "spark", "orbit ring"][: 2 + int(trait_scores["openness"] > 0.55)],
    )


def build_exemplars(messages: list[str], limit: int = 4) -> list[ExemplarPair]:
    selected = sorted(messages, key=len, reverse=True)[:limit]
    pairs: list[ExemplarPair] = []
    for msg in selected:
        neutral = re.sub(r"[!?]+", ".", msg)
        neutral = re.sub(r"\b(I'm|I'm|I've|I'll|I'd)\b", "They are", neutral, flags=re.I)
        neutral = re.sub(r"\b(I|me|my|mine)\b", "they", neutral, flags=re.I)
        if neutral.strip().lower() == msg.strip().lower():
            neutral = f"A neutral restatement: {msg[:120]}"
        pairs.append(ExemplarPair(style_example=msg, neutral_contrast=neutral))
    return pairs
