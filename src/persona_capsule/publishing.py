"""Explicit public projections, social metadata, and reversible publishing."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from html import escape
from pathlib import Path
from secrets import token_urlsafe
from typing import Any
from urllib.parse import quote

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.repository import (
    CapsuleNotFoundError,
    CapsuleRecord,
    CapsuleRepository,
)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class PublishSelection:
    include_summary: bool = True
    include_descriptors: bool = True
    include_dimensions: bool = False
    include_card: bool = True
    include_voice_sample: bool = False


class PublishingService:
    def __init__(
        self,
        capsule_library: CapsuleLibrary,
        repository: CapsuleRepository,
        artifact_root: str | Path,
        public_base_url: str,
    ) -> None:
        self._capsule_library = capsule_library
        self._repository = repository
        self._artifact_root = Path(artifact_root)
        self._public_base_url = public_base_url.rstrip("/")

    def preview(
        self,
        principal: Principal | None,
        capsule_id: str,
        selection: PublishSelection,
    ) -> dict[str, Any]:
        record = self._capsule_library.get_capsule(principal, capsule_id)
        if record.public_projection is None:
            raise ValueError("An approved public-safe profile is required.")
        projection: dict[str, Any] = {
            "name": record.name,
            "generated_art_label": "AI-generated artwork",
        }
        if selection.include_summary:
            projection["summary"] = record.public_projection.summary
        if selection.include_descriptors:
            projection["descriptors"] = list(record.public_projection.descriptors)
        if selection.include_dimensions:
            projection["dimensions"] = dict(record.public_projection.dimensions)
        if selection.include_card:
            if not record.social_image_ref:
                raise ValueError("Generate a social card before publishing it.")
            projection["social_image"] = True
        if selection.include_voice_sample:
            if not record.voice_sample_ref or record.voice_status != "ready":
                raise ValueError("Create an available voice sample before publishing it.")
            projection["voice_sample"] = True
            projection["voice_label"] = "Synthetic voice generated with ElevenLabs"
        return projection

    def publish(
        self,
        principal: Principal | None,
        capsule_id: str,
        selection: PublishSelection,
        *,
        confirmed: bool,
    ) -> CapsuleRecord:
        if not confirmed:
            raise ValueError("Confirm the field-level public preview before publishing.")
        record = self._capsule_library.get_capsule(principal, capsule_id)
        projection = self.preview(principal, capsule_id, selection)
        slug = record.public_slug or token_urlsafe(18)
        return self._capsule_library.save_capsule(
            principal,
            replace(
                record,
                is_published=True,
                public_slug=slug,
                published_projection=projection,
                published_at=_now_iso(),
            ),
        )

    def unpublish(
        self,
        principal: Principal | None,
        capsule_id: str,
    ) -> CapsuleRecord:
        record = self._capsule_library.get_capsule(principal, capsule_id)
        return self._capsule_library.save_capsule(
            principal,
            replace(
                record,
                is_published=False,
                published_projection=None,
                published_at="",
            ),
        )

    def public_url(self, record: CapsuleRecord) -> str:
        if not record.is_published or not record.public_slug:
            raise ValueError("Capsule is not published.")
        return f"{self._public_base_url}/c/{record.public_slug}"

    def x_share_url(self, record: CapsuleRecord) -> str:
        url = self.public_url(record)
        text = f"Meet {record.name}, my Persona Capsule."
        return f"https://x.com/intent/post?text={quote(text)}&url={quote(url)}"

    def get_public(self, slug: str) -> CapsuleRecord:
        if not slug or len(slug) > 64:
            raise CapsuleNotFoundError(slug)
        record = self._repository.get_public_by_slug(slug)
        if record.published_projection is None:
            raise CapsuleNotFoundError(slug)
        return record

    def public_image_path(self, record: CapsuleRecord) -> Path:
        projection = record.published_projection or {}
        if not projection.get("social_image") or not record.social_image_ref:
            raise CapsuleNotFoundError(record.public_slug)
        owner_namespace = sha256(record.owner_id.encode()).hexdigest()[:24]
        path = (
            self._artifact_root
            / "artifacts"
            / owner_namespace
            / record.capsule_id
            / Path(record.social_image_ref).name
        )
        if not path.is_file():
            raise CapsuleNotFoundError(record.public_slug)
        return path

    def public_audio_path(self, record: CapsuleRecord) -> Path:
        projection = record.published_projection or {}
        if not projection.get("voice_sample") or not record.voice_sample_ref:
            raise CapsuleNotFoundError(record.public_slug)
        owner_namespace = sha256(record.owner_id.encode()).hexdigest()[:24]
        path = (
            self._artifact_root
            / "artifacts"
            / owner_namespace
            / record.capsule_id
            / Path(record.voice_sample_ref).name
        )
        if not path.is_file():
            raise CapsuleNotFoundError(record.public_slug)
        return path

    def render_public_html(self, record: CapsuleRecord) -> str:
        projection = record.published_projection or {}
        title = str(projection["name"])
        summary = str(
            projection.get(
                "summary",
                "A collectible communication-style Persona Capsule.",
            )
        )
        canonical = self.public_url(record)
        image_url = (
            f"{canonical}/image"
            if projection.get("social_image")
            else f"{self._public_base_url}/app/"
        )
        descriptors = projection.get("descriptors", [])
        descriptors_html = "".join(
            f"<span>{escape(str(descriptor))}</span>" for descriptor in descriptors
        )
        dimensions = projection.get("dimensions", {})
        dimensions_html = "".join(
            f"<li><b>{escape(str(name).replace('_', ' ').title())}</b>"
            f"<span>{float(value):.0f}</span></li>"
            for name, value in dimensions.items()
        )
        share_url = self.x_share_url(record)
        image_html = (
            f'<img src="{escape(image_url)}" alt="{escape(title)} generated card">'
            if projection.get("social_image")
            else ""
        )
        voice_html = (
            '<div class="voice"><b>Synthetic voice</b>'
            f'<audio controls preload="none" src="{escape(canonical)}/audio"></audio>'
            f"<small>{escape(str(projection.get('voice_label', 'Synthetic audio')))}</small></div>"
            if projection.get("voice_sample")
            else ""
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} · Persona Capsule</title>
  <meta name="description" content="{escape(summary)}">
  <link rel="canonical" href="{escape(canonical)}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{escape(title)} · Persona Capsule">
  <meta property="og:description" content="{escape(summary)}">
  <meta property="og:url" content="{escape(canonical)}">
  <meta property="og:image" content="{escape(image_url)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{escape(title)} · Persona Capsule">
  <meta name="twitter:description" content="{escape(summary)}">
  <meta name="twitter:image" content="{escape(image_url)}">
  <style>
    :root {{ --paper:#f3efe2; --ink:#171712; --rust:#d4512d; --acid:#d8f24a; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink);
      font-family:Arial,sans-serif; }}
    main {{ max-width:1120px; margin:auto; padding:32px 24px 80px; }}
    nav {{ border-bottom:1px solid var(--ink); padding:12px 0; font-weight:800; }}
    article {{ background:var(--ink); color:var(--paper); margin-top:56px; }}
    article img {{ display:block; width:100%; aspect-ratio:1200/628; object-fit:cover;
      border-bottom:1px solid #555; }}
    section {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(240px,320px);
      gap:64px; padding:56px; }}
    section.no-dimensions {{ grid-template-columns:1fr; }}
    h1 {{ font:500 clamp(48px,8vw,92px)/.9 Georgia,serif; letter-spacing:-.05em; }}
    p {{ font-size:21px; line-height:1.5; }}
    .traits {{ display:flex; flex-wrap:wrap; gap:8px; margin:28px 0; }}
    .traits span {{ border:1px solid #777; border-radius:999px; padding:8px 12px; }}
    ul {{ list-style:none; padding:0; }}
    li {{ display:flex; justify-content:space-between; border-top:1px solid #555;
      padding:12px 0; }}
    a {{ display:inline-block; background:var(--acid); color:var(--ink);
      padding:14px 18px; font-weight:800; text-decoration:none; margin-top:24px; }}
    small {{ display:block; margin-top:24px; opacity:.7; }}
    .voice {{ border-top:1px solid #555; margin-top:28px; padding-top:20px; }}
    .voice audio {{ display:block; margin-top:12px; max-width:100%; }}
    .voice small {{ margin-top:8px; }}
    @media(max-width:760px) {{ section {{ grid-template-columns:1fr; padding:32px; }} }}
  </style>
</head>
<body><main>
  <nav>PERSONA CAPSULE / PUBLIC PROJECTION</nav>
  <article>
    {image_html}
    <section class="{"with-dimensions" if dimensions_html else "no-dimensions"}">
      <div>
        <h1>{escape(title)}</h1>
        <p>{escape(summary)}</p>
        <div class="traits">{descriptors_html}</div>
        {voice_html}
        <a href="{escape(share_url)}" rel="noopener">Share on X</a>
        <small>Communication-style collectible · AI-generated artwork</small>
      </div>
      {f"<ul>{dimensions_html}</ul>" if dimensions_html else ""}
    </section>
  </article>
</main></body></html>"""
