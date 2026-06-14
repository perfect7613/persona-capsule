"""Message parsing, validation, redaction, and deterministic style profiling."""

import re
from dataclasses import dataclass
from hashlib import sha256

from persona_capsule.profile import (
    ApprovedProfile,
    ExemplarPair,
    StyleDimensions,
    StyleProfile,
    ensure_distinct_contrast,
)

MIN_MESSAGES = 8
MIN_UNIQUE_MESSAGES = 6
MIN_TOTAL_CHARACTERS = 240
MAX_RETAINED_PAIRS = 4

_SPEAKER_LINE = re.compile(r"^(?:\[[^\]]{1,40}\]\s*)?(?P<author>[\w .@-]{1,40}):\s*(?P<text>.+)$")
_REDACTIONS = (
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("URL", re.compile(r"\bhttps?://[^\s<>()]+|\bwww\.[^\s<>()]+", re.I)),
    (
        "PHONE",
        re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)"),
    ),
    (
        "IDENTIFIER",
        re.compile(
            r"\b(?:account|customer|member|order|ticket|user)[\s_-]*(?:id|no|number)?"
            r"[\s:#-]*[A-Z0-9][A-Z0-9_-]{3,}\b",
            re.I,
        ),
    ),
)
_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "i'm": "I am",
    "i've": "I have",
    "i'll": "I will",
    "we're": "we are",
    "we've": "we have",
    "that's": "that is",
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
}


class IngestionError(ValueError):
    """Actionable input or consent failure."""


@dataclass(frozen=True, slots=True)
class MessageRecord:
    author: str
    text: str


@dataclass(frozen=True, slots=True)
class Redaction:
    kind: str
    original: str
    replacement: str


@dataclass(frozen=True, slots=True)
class IngestionDraft:
    raw_input: str
    speaker: str
    messages: tuple[MessageRecord, ...]
    redactions: tuple[Redaction, ...]
    profile: StyleProfile
    proposed_pairs: tuple[ExemplarPair, ...]
    source_fingerprint: str

    def discard_raw(self) -> "IngestionDraft":
        return IngestionDraft(
            raw_input="",
            speaker=self.speaker,
            messages=(),
            redactions=(),
            profile=self.profile,
            proposed_pairs=self.proposed_pairs,
            source_fingerprint=self.source_fingerprint,
        )


def parse_messages(raw_input: str, default_author: str = "You") -> tuple[MessageRecord, ...]:
    normalized = raw_input.replace("\r\n", "\n").replace("\r", "\n")
    records: list[MessageRecord] = []
    for line in normalized.splitlines():
        clean = line.strip().lstrip("-•").strip()
        if not clean:
            continue
        match = _SPEAKER_LINE.match(clean)
        if match:
            records.append(
                MessageRecord(
                    author=match.group("author").strip(),
                    text=match.group("text").strip(),
                )
            )
        else:
            records.append(MessageRecord(author=default_author, text=clean))
    if not records:
        raise IngestionError("Paste at least eight non-empty messages, one per line.")
    return tuple(records)


def select_speaker(
    messages: tuple[MessageRecord, ...],
    speaker: str,
) -> tuple[MessageRecord, ...]:
    selected = tuple(
        message for message in messages if message.author.casefold() == speaker.strip().casefold()
    )
    if not selected:
        known = ", ".join(sorted({message.author for message in messages}))
        raise IngestionError(
            f'No messages matched speaker "{speaker}". Detected speakers: {known}.'
        )
    return selected


def validate_sample(messages: tuple[MessageRecord, ...]) -> None:
    if len(messages) < MIN_MESSAGES:
        raise IngestionError(
            f"Only {len(messages)} messages matched. Provide at least {MIN_MESSAGES}; "
            "around 20 varied messages works best."
        )
    unique = {re.sub(r"\s+", " ", message.text).casefold() for message in messages}
    if len(unique) < MIN_UNIQUE_MESSAGES:
        raise IngestionError(
            "The sample is too repetitive. Provide at least six meaningfully different messages."
        )
    character_count = sum(len(message.text) for message in messages)
    if character_count < MIN_TOTAL_CHARACTERS:
        raise IngestionError(
            f"The selected sample has {character_count} characters. Provide at least "
            f"{MIN_TOTAL_CHARACTERS} characters so cadence and vocabulary are observable."
        )


def redact_messages(
    messages: tuple[MessageRecord, ...],
) -> tuple[tuple[MessageRecord, ...], tuple[Redaction, ...]]:
    redactions: list[Redaction] = []
    cleaned: list[MessageRecord] = []
    counters: dict[str, int] = {}

    for message in messages:
        text = message.text
        for kind, pattern in _REDACTIONS:

            def replace_match(match: re.Match[str], *, kind: str = kind) -> str:
                counters[kind] = counters.get(kind, 0) + 1
                replacement = f"[{kind}_{counters[kind]}]"
                redactions.append(Redaction(kind, match.group(0), replacement))
                return replacement

            text = pattern.sub(replace_match, text)
        cleaned.append(MessageRecord(author=message.author, text=text))
    return tuple(cleaned), tuple(redactions)


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def infer_style_profile(messages: tuple[MessageRecord, ...]) -> StyleProfile:
    texts = [message.text for message in messages]
    joined = " ".join(texts)
    words = re.findall(r"\b[\w'-]+\b", joined)
    word_count = max(1, len(words))
    avg_words = sum(len(re.findall(r"\b[\w'-]+\b", text)) for text in texts) / len(texts)
    questions = joined.count("?") / len(texts)
    exclamations = joined.count("!") / len(texts)
    contractions = len(re.findall(r"\b\w+'\w+\b", joined)) / word_count
    warmth_terms = (
        len(
            re.findall(
                r"\b(?:thanks|thank you|please|appreciate|glad|happy|we|together)\b", joined, re.I
            )
        )
        / word_count
    )
    structure_terms = (
        len(
            re.findall(r"\b(?:first|next|then|finally|because|therefore|step|plan)\b", joined, re.I)
        )
        / word_count
    )
    hedges = (
        len(re.findall(r"\b(?:maybe|perhaps|might|could|I think|seems)\b", joined, re.I))
        / word_count
    )
    novel_words = len({word.casefold() for word in words}) / word_count

    dimensions = StyleDimensions(
        openness=_clamp(35 + novel_words * 80 + questions * 12),
        conscientiousness=_clamp(42 + structure_terms * 900 + min(avg_words, 24)),
        expressiveness=_clamp(32 + exclamations * 35 + contractions * 500),
        agreeableness=_clamp(38 + warmth_terms * 1100 + hedges * 350),
        emotional_range=_clamp(35 + exclamations * 28 + questions * 12),
        directness=_clamp(74 - hedges * 800 - questions * 10),
        formality=_clamp(72 - contractions * 1200 - exclamations * 18),
    )
    descriptors = []
    descriptors.append("direct" if dimensions.directness >= 58 else "deliberative")
    descriptors.append("warm" if dimensions.agreeableness >= 56 else "reserved")
    descriptors.append("expressive" if dimensions.expressiveness >= 58 else "measured")
    descriptors.append("structured" if dimensions.conscientiousness >= 58 else "fluid")
    rhythm = (
        "Compact sentences with quick pivots."
        if avg_words < 12
        else "Medium-length sentences with room for qualification."
        if avg_words < 22
        else "Long-form sentences with layered detail."
    )
    tendencies = []
    if contractions > 0.01:
        tendencies.append("uses conversational contractions")
    if questions > 0.2:
        tendencies.append("uses questions to open or refine ideas")
    if structure_terms > 0.008:
        tendencies.append("signals sequence and causality")
    if exclamations > 0.15:
        tendencies.append("uses emphatic punctuation")
    if not tendencies:
        tendencies.append("prefers plain declarative phrasing")

    evidence = tuple(sorted(texts, key=lambda text: abs(len(text) - 110))[:3])
    uncertainty = round(max(0.08, 0.52 - min(len(messages), 24) * 0.018), 2)
    return StyleProfile(
        summary=(
            f"A {descriptors[0]}, {descriptors[1]} communicator with a "
            f"{descriptors[2]} and {descriptors[3]} delivery."
        ),
        descriptors=tuple(descriptors),
        lexical_tendencies=tuple(tendencies),
        sentence_rhythm=rhythm,
        dimensions=dimensions,
        evidence=evidence,
        uncertainty=uncertainty,
    )


def neutralize_style(text: str) -> str:
    neutral = text
    for contraction, expanded in _CONTRACTIONS.items():
        neutral = re.sub(re.escape(contraction), expanded, neutral, flags=re.I)
    neutral = re.sub(r"[!?]{2,}", ".", neutral)
    neutral = re.sub(r"!+", ".", neutral)
    neutral = re.sub(r"\s*[—–]\s*", ", ", neutral)
    neutral = re.sub(r"^\s*please\s+", "", neutral, flags=re.I)
    neutral = re.sub(r"^\s*(?:thanks|thank you)\s+for\s+", "", neutral, flags=re.I)
    neutral = re.sub(r"^\s*good progress[.!]\s*", "", neutral, flags=re.I)
    neutral = re.sub(r"\blet[’']s\b", "we should", neutral, flags=re.I)
    neutral = re.sub(
        r"\b(?:really|very|honestly|basically|literally)\b\s*", "", neutral, flags=re.I
    )
    neutral = re.sub(r"\s+", " ", neutral).strip()
    return ensure_distinct_contrast(text, neutral)[0]


def propose_exemplar_pairs(messages: tuple[MessageRecord, ...]) -> tuple[ExemplarPair, ...]:
    ranked = sorted(
        enumerate(messages),
        key=lambda item: (
            -len(set(re.findall(r"\b\w+\b", item[1].text.casefold()))),
            -len(item[1].text),
            item[0],
        ),
    )
    selected = sorted(ranked[:MAX_RETAINED_PAIRS], key=lambda item: item[0])
    return tuple(
        ExemplarPair(
            positive=message.text,
            neutral=neutralize_style(message.text),
            source_index=index,
        )
        for index, message in selected
    )


def build_ingestion_draft(
    raw_input: str,
    speaker: str,
    consent: bool,
) -> IngestionDraft:
    if not consent:
        raise IngestionError(
            "Confirm that you own these messages or have permission to process them."
        )
    parsed = parse_messages(raw_input)
    selected = select_speaker(parsed, speaker)
    validate_sample(selected)
    cleaned, redactions = redact_messages(selected)
    profile = infer_style_profile(cleaned)
    pairs = propose_exemplar_pairs(cleaned)
    fingerprint = sha256(raw_input.encode()).hexdigest()
    return IngestionDraft(
        raw_input=raw_input,
        speaker=speaker.strip(),
        messages=cleaned,
        redactions=redactions,
        profile=profile,
        proposed_pairs=pairs,
        source_fingerprint=fingerprint,
    )


def approve_draft(
    draft: IngestionDraft,
    selected_pair_hashes: set[str],
    dimension_edits: dict[str, float],
) -> ApprovedProfile:
    selected = tuple(
        pair for pair in draft.proposed_pairs if pair.pair_hash in selected_pair_hashes
    )
    if not selected:
        raise IngestionError("Approve at least one exemplar and neutral contrast pair.")
    profile = StyleProfile(
        summary=draft.profile.summary,
        descriptors=draft.profile.descriptors,
        lexical_tendencies=draft.profile.lexical_tendencies,
        sentence_rhythm=draft.profile.sentence_rhythm,
        dimensions=draft.profile.dimensions.edited(dimension_edits),
        evidence=draft.profile.evidence,
        uncertainty=draft.profile.uncertainty,
    )
    return ApprovedProfile(
        profile=profile,
        exemplar_pairs=selected,
        source_fingerprint=draft.source_fingerprint,
    )
