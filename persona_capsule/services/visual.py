"""Card art and social preview rendering."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from persona_capsule.models.capsule import CapsuleRecord


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_card(canvas: Image.Image, capsule: CapsuleRecord, title_suffix: str) -> None:
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    profile = capsule.profile

    draw.rectangle((0, 0, width, height), fill=(18, 24, 38))
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=28, fill=(32, 42, 68))
    draw.rounded_rectangle((40, 40, width - 40, 140), radius=18, fill=(255, 176, 84))

    title_font = _load_font(42 if height >= 600 else 28)
    body_font = _load_font(24 if height >= 600 else 18)
    small_font = _load_font(18 if height >= 600 else 14)

    draw.text((56, 58), f"{capsule.display_name}{title_suffix}", fill=(20, 20, 20), font=title_font)
    draw.text((56, 170), profile.summary[:180], fill=(230, 236, 245), font=body_font)

    y = 260
    for trait in profile.traits[:3]:
        bar_width = int(280 * trait.score)
        draw.text((56, y), f"{trait.name.title()} · {trait.label}", fill=(180, 190, 210), font=small_font)
        draw.rounded_rectangle((56, y + 24, 56 + bar_width, y + 36), radius=6, fill=(120, 196, 255))
        y += 56

    draw.text((56, height - 72), f"Palette: {profile.palette}", fill=(255, 210, 140), font=small_font)
    draw.text((56, height - 44), "Persona Capsule · communication-style collectible", fill=(150, 160, 180), font=small_font)


def render_card_image(capsule: CapsuleRecord, output_path: Path) -> Path:
    image = Image.new("RGB", (768, 1024), color=(18, 24, 38))
    _draw_card(image, capsule, "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def render_share_image(capsule: CapsuleRecord, output_path: Path) -> Path:
    image = Image.new("RGB", (1200, 628), color=(18, 24, 38))
    _draw_card(image, capsule, " · Share Card")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path
