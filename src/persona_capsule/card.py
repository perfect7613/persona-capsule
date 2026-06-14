"""Profile-safe card prompting, art providers, and deterministic composition."""

import io
import random
import textwrap
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.repository import CapsuleRecord

INTERACTIVE_SIZE = (768, 1024)
SOCIAL_SIZE = (1200, 628)
ART_SIZE = (768, 768)
VARIATIONS = {
    "signal": "centered monolithic composition, precise radial focus",
    "archive": "layered archival specimen composition, quiet asymmetry",
    "kinetic": "diagonal kinetic composition, controlled motion and energy",
}
ANIME_STYLE_TRIGGER = "anime_style"
ANIME_TRAIT_CUES = {
    "analytical": "observant eyes and a thoughtful, composed expression",
    "calm": "relaxed posture and a serene expression",
    "concise": "a clean silhouette and an economical, confident pose",
    "curious": "an attentive gaze and a sense of discovery",
    "direct": "a focused gaze and decisive posture",
    "expressive": "animated eyes and an energetic, readable expression",
    "measured": "calm posture and a restrained, deliberate expression",
    "playful": "a lively pose and a subtle mischievous smile",
    "structured": "precise tailoring and orderly geometric accessories",
    "warm": "gentle eyes and an approachable, open expression",
}
ANIME_SIGNATURE_HAIR = (
    "short textured hair with a clean side part",
    "layered swept-back hair",
    "an asymmetric chin-length haircut",
    "long hair tied into a precise low ponytail",
    "a sharp cropped bob",
    "soft wavy hair with one distinctive forelock",
)
ANIME_SIGNATURE_ACCESSORIES = (
    "a geometric ear cuff",
    "thin rectangular glasses",
    "a minimal enamel lapel pin",
    "a translucent collar clasp",
    "a structured scarf",
    "one asymmetric metallic shoulder detail",
)
ANIME_SIGNATURE_AURAS = (
    "an offset halo made from broken rings",
    "a constellation of small modular tiles",
    "a narrow luminous horizon crossed by two arcs",
    "a stepped waveform made from clean light bars",
    "an orbital map with one missing segment",
    "a layered fan of translucent geometric panels",
)
PALETTES = (
    ((19, 18, 16), (216, 242, 74), (212, 81, 45), (243, 239, 226)),
    ((14, 23, 38), (80, 210, 196), (242, 165, 65), (235, 239, 245)),
    ((30, 15, 37), (231, 92, 155), (128, 225, 190), (248, 231, 201)),
    ((18, 28, 22), (190, 221, 112), (231, 117, 76), (239, 232, 210)),
)
SYMBOLS = {
    "analytical": "a calibrated orbital grid",
    "direct": "a single precision beam",
    "warm": "a glowing open aperture",
    "playful": "a restrained chromatic spring",
    "concise": "a clean geometric cut",
    "curious": "an unfolding cartographic contour",
    "expressive": "a rhythmic field of marks",
    "calm": "a balanced horizon",
}


def _font(size: int, *, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ("Georgia.ttf", "DejaVuSerif.ttf") if serif else ("Avenir Next.ttc", "DejaVuSans.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True, slots=True)
class CardPrompt:
    text: str
    palette: tuple[tuple[int, int, int], ...]
    variation: str
    seed: int

    @property
    def prompt_hash(self) -> str:
        return sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtResult:
    png_bytes: bytes
    provider: str
    model_id: str
    model_revision: str


@dataclass(frozen=True, slots=True)
class CardResult:
    record: CapsuleRecord
    interactive_path: Path
    social_path: Path
    provider: str
    used_fallback: bool
    prompt: CardPrompt


class ArtProvider(Protocol):
    def generate(self, prompt: CardPrompt) -> ArtResult: ...


def build_card_prompt(
    record: CapsuleRecord,
    *,
    variation: str,
    seed: int,
) -> CardPrompt:
    if record.style_profile is None or record.public_projection is None:
        raise ValueError("An approved public-safe profile is required for card generation.")
    if variation not in VARIATIONS:
        raise ValueError(f"Unknown visual variation: {variation}")
    dimensions = record.style_profile.dimensions
    palette_index = int(
        (dimensions.openness + dimensions.expressiveness + dimensions.formality) // 75
    ) % len(PALETTES)
    palette = PALETTES[palette_index]
    mapped_symbols = [
        SYMBOLS[descriptor.casefold()]
        for descriptor in record.public_projection.descriptors
        if descriptor.casefold() in SYMBOLS
    ][:3]
    if not mapped_symbols:
        mapped_symbols = ["an abstract signal glyph", "a measured concentric field"]
    descriptors = [descriptor.casefold() for descriptor in record.public_projection.descriptors]
    trait_cues = [
        ANIME_TRAIT_CUES[descriptor] for descriptor in descriptors if descriptor in ANIME_TRAIT_CUES
    ]
    signature_source = "|".join(
        (
            record.capsule_id,
            ",".join(descriptors),
            ",".join(
                f"{name}:{value:.1f}"
                for name, value in sorted(record.public_projection.dimensions.items())
            ),
        )
    )
    signature_digest = sha256(signature_source.encode()).digest()
    signature_hair = ANIME_SIGNATURE_HAIR[signature_digest[0] % len(ANIME_SIGNATURE_HAIR)]
    signature_accessory = ANIME_SIGNATURE_ACCESSORIES[
        signature_digest[1] % len(ANIME_SIGNATURE_ACCESSORIES)
    ]
    signature_aura = ANIME_SIGNATURE_AURAS[signature_digest[2] % len(ANIME_SIGNATURE_AURAS)]
    energy = (
        "high-energy but ordered" if dimensions.expressiveness >= 65 else "quiet and deliberate"
    )
    geometry = "sharp modular geometry" if dimensions.directness >= 65 else "soft layered geometry"
    finish = (
        "formal editorial precision"
        if dimensions.formality >= 60
        else "tactile independent-publishing character"
    )
    prompt = (
        f"{ANIME_STYLE_TRIGGER}, masterpiece, best quality, premium contemporary anime "
        "illustration. One original fictional adult anime character, solo, waist-up portrait, "
        "face clearly visible, expressive eyes, polished character design, cinematic lighting, "
        "and no resemblance to a real person. "
        f"The character embodies these personality traits: {', '.join(descriptors)}. "
        f"Express those traits through {', '.join(trait_cues) or 'a distinctive presence'}. "
        f"Stable capsule signature: {signature_hair}, {signature_accessory}, and "
        f"{signature_aura}. "
        f"{VARIATIONS[variation]}. {energy}; {geometry}; {finish}. "
        f"Keep these motifs as subtle background or aura elements only: "
        f"{', '.join(mapped_symbols)}. "
        "Collectible social profile card, controlled cel shading, crisp linework, rich color, "
        "dramatic negative space. No words, no letters, no captions, no logos, no watermark, "
        "no photorealism."
    )
    return CardPrompt(
        text=prompt,
        palette=palette,
        variation=variation,
        seed=int(seed),
    )


def _palette_for_record(
    record: CapsuleRecord,
) -> tuple[tuple[int, int, int], ...]:
    if record.style_profile is None:
        raise ValueError("An approved profile is required.")
    dimensions = record.style_profile.dimensions
    palette_index = int(
        (dimensions.openness + dimensions.expressiveness + dimensions.formality) // 75
    ) % len(PALETTES)
    return PALETTES[palette_index]


class DeterministicArtProvider:
    """Generate coherent local art when the GPU provider is unavailable."""

    def generate(self, prompt: CardPrompt) -> ArtResult:
        rng = random.Random(prompt.seed)
        dark, accent, rust, paper = prompt.palette
        image = Image.new("RGB", ART_SIZE, dark)
        draw = ImageDraw.Draw(image, "RGBA")

        for index in range(18):
            radius = rng.randint(18, 170)
            x = rng.randint(-80, ART_SIZE[0] + 80)
            y = rng.randint(-80, ART_SIZE[1] + 80)
            color = (accent, rust, paper)[index % 3]
            alpha = rng.randint(28, 115)
            if prompt.variation == "archive":
                draw.rounded_rectangle(
                    (x - radius, y - radius // 2, x + radius, y + radius // 2),
                    radius=12,
                    outline=(*color, alpha),
                    width=rng.randint(2, 9),
                )
            elif prompt.variation == "kinetic":
                draw.polygon(
                    (
                        (x - radius, y + radius),
                        (x + radius, y - radius),
                        (x + radius // 2, y + radius),
                    ),
                    fill=(*color, alpha),
                )
            else:
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=(*color, alpha),
                    width=rng.randint(3, 12),
                )

        center = (ART_SIZE[0] // 2, ART_SIZE[1] // 2)
        for radius, color in ((180, rust), (120, accent), (54, paper)):
            draw.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ),
                fill=(*color, 210 if radius == 54 else 48),
                outline=(*color, 230),
                width=5,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return ArtResult(
            png_bytes=buffer.getvalue(),
            provider="deterministic-fallback",
            model_id="local-geometric-renderer",
            model_revision="persona-card-v1",
        )


def _fit_art(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    source = image.convert("RGB")
    ratio = max(size[0] / source.width, size[1] / source.height)
    resized = source.resize(
        (round(source.width * ratio), round(source.height * ratio)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def _wrapped(draw: ImageDraw.ImageDraw, text: str, width: int) -> str:
    del draw
    return "\n".join(textwrap.wrap(text, width=width, max_lines=3, placeholder="..."))


def render_interactive_card(record: CapsuleRecord, art: Image.Image) -> Image.Image:
    if record.public_projection is None:
        raise ValueError("Public projection required.")
    dark, accent, rust, paper = _palette_for_record(record)
    canvas = Image.new("RGB", INTERACTIVE_SIZE, paper)
    canvas.paste(_fit_art(art, (INTERACTIVE_SIZE[0], 620)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 620, 768, 1024), fill=dark)
    draw.rectangle((32, 32, 736, 992), outline=paper, width=2)
    draw.rectangle((44, 588, 724, 654), fill=accent)
    draw.text((62, 602), "PERSONA CAPSULE / PRIVATE EDITION", fill=dark, font=_font(16))
    draw.text((54, 690), record.name, fill=paper, font=_font(54, serif=True))
    summary = _wrapped(draw, record.public_projection.summary, 48)
    draw.multiline_text(
        (56, 770),
        summary,
        fill=paper,
        font=_font(22),
        spacing=8,
    )
    descriptor_text = "  /  ".join(
        descriptor.upper() for descriptor in record.public_projection.descriptors[:4]
    )
    draw.text((56, 944), descriptor_text, fill=rust, font=_font(15))
    return canvas


def render_social_card(record: CapsuleRecord, art: Image.Image) -> Image.Image:
    if record.public_projection is None:
        raise ValueError("Public projection required.")
    dark, accent, rust, paper = _palette_for_record(record)
    canvas = Image.new("RGB", SOCIAL_SIZE, dark)
    canvas.paste(_fit_art(art, (560, 628)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((560, 0, 1200, 628), fill=dark)
    draw.rectangle((584, 24, 1176, 604), outline=paper, width=2)
    draw.rectangle((608, 54, 1138, 91), fill=accent)
    draw.text((620, 61), "PERSONA CAPSULE / SIGNAL OBJECT", fill=dark, font=_font(14))
    draw.text((610, 145), record.name, fill=paper, font=_font(58, serif=True))
    summary = _wrapped(draw, record.public_projection.summary, 40)
    draw.multiline_text(
        (612, 245),
        summary,
        fill=paper,
        font=_font(23),
        spacing=10,
    )
    draw.line((612, 488, 1138, 488), fill=rust, width=4)
    descriptors = "  ".join(
        descriptor.upper() for descriptor in record.public_projection.descriptors[:4]
    )
    draw.text((612, 520), descriptors, fill=accent, font=_font(16))
    draw.text((612, 563), "COMMUNICATION STYLE / AI-GENERATED ART", fill=paper, font=_font(13))
    return canvas


class CapsuleCardService:
    def __init__(
        self,
        capsule_library: CapsuleLibrary,
        artifact_root: str | Path,
        primary_provider: ArtProvider | None = None,
    ) -> None:
        self._capsule_library = capsule_library
        self._artifact_root = Path(artifact_root)
        self._primary_provider = primary_provider
        self._fallback = DeterministicArtProvider()

    def generate(
        self,
        principal: Principal | None,
        capsule_id: str,
        *,
        variation: str,
        seed: int,
    ) -> CardResult:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        record = self._capsule_library.get_capsule(principal, capsule_id)
        prompt = build_card_prompt(record, variation=variation, seed=seed)
        used_fallback = self._primary_provider is None
        if self._primary_provider is None:
            art_result = self._fallback.generate(prompt)
        else:
            try:
                art_result = self._primary_provider.generate(prompt)
            except Exception:
                art_result = self._fallback.generate(prompt)
                used_fallback = True

        art = Image.open(io.BytesIO(art_result.png_bytes))
        interactive = render_interactive_card(record, art)
        social = render_social_card(record, art)
        owner_namespace = sha256(principal.user_id.encode()).hexdigest()[:24]
        output_dir = self._artifact_root / "artifacts" / owner_namespace / record.capsule_id
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"card-{variation}-{int(seed)}"
        interactive_path = output_dir / f"{stem}-interactive.png"
        social_path = output_dir / f"{stem}-social.png"
        interactive.save(interactive_path, format="PNG", optimize=True)
        social.save(social_path, format="PNG", optimize=True)
        interactive_path.chmod(0o600)
        social_path.chmod(0o600)

        relative_interactive = f"card/{interactive_path.name}"
        relative_social = f"card/{social_path.name}"
        prior_artifacts = tuple(
            reference for reference in record.artifact_refs if not reference.startswith("card/")
        )
        updated = self._capsule_library.save_capsule(
            principal,
            replace(
                record,
                artifact_refs=prior_artifacts + (relative_interactive, relative_social),
                card_image_ref=relative_interactive,
                social_image_ref=relative_social,
                card_seed=int(seed),
                card_prompt_hash=prompt.prompt_hash,
                card_provider=art_result.provider,
                card_model_id=art_result.model_id,
                card_model_revision=art_result.model_revision,
            ),
        )
        return CardResult(
            record=updated,
            interactive_path=interactive_path,
            social_path=social_path,
            provider=art_result.provider,
            used_fallback=used_fallback,
            prompt=prompt,
        )
