"""Custom Gradio interface for Persona Capsule."""

from hashlib import sha256
from html import escape
from pathlib import Path
from uuid import uuid4

import gradio as gr

from persona_capsule.card import CapsuleCardService
from persona_capsule.config import Settings
from persona_capsule.demo import DEMO_CAPSULE, demo_reply
from persona_capsule.export import build_capsule_export
from persona_capsule.identity import IdentityGateway, Principal
from persona_capsule.ingestion import (
    IngestionDraft,
    IngestionError,
    approve_draft,
    build_ingestion_draft,
)
from persona_capsule.library import CapsuleLibrary
from persona_capsule.publishing import PublishingService, PublishSelection
from persona_capsule.repository import CapsuleRecord
from persona_capsule.steering import SteeringError
from persona_capsule.steering_service import CapsuleSteeringService

CSS = """
:root {
  --paper: #f3efe2;
  --ink: #171712;
  --rust: #d4512d;
  --acid: #d8f24a;
  --rule: rgba(23, 23, 18, 0.22);
}

body, .gradio-container {
  background:
    radial-gradient(circle at 18% 8%, rgba(216, 242, 74, 0.28), transparent 24rem),
    linear-gradient(90deg, rgba(23, 23, 18, 0.035) 1px, transparent 1px),
    linear-gradient(rgba(23, 23, 18, 0.035) 1px, transparent 1px),
    var(--paper) !important;
  background-size: auto, 32px 32px, 32px 32px, auto !important;
  color: var(--ink) !important;
  font-family: "Avenir Next", "Gill Sans", sans-serif !important;
}

.gradio-container {
  max-width: 1240px !important;
  padding: 22px 24px 72px !important;
}

.pc-kicker, .pc-label {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.17em;
  text-transform: uppercase;
}

.pc-nav {
  align-items: center;
  border-bottom: 1px solid var(--ink);
  display: flex;
  justify-content: space-between;
  padding: 8px 0 14px;
}

.pc-wordmark {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 22px;
  font-weight: 700;
}

.pc-hero {
  display: grid;
  gap: 36px;
  grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.55fr);
  padding: 70px 0 52px;
}

.pc-hero h1 {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(54px, 8.6vw, 118px);
  font-weight: 500;
  letter-spacing: -0.065em;
  line-height: 0.84;
  margin: 18px 0 26px;
  max-width: 900px;
}

.pc-hero h1 em {
  color: var(--rust);
  font-weight: 500;
}

.pc-deck {
  font-size: clamp(17px, 2vw, 23px);
  line-height: 1.45;
  max-width: 660px;
}

.pc-side-note {
  align-self: end;
  border-left: 1px solid var(--ink);
  padding: 10px 0 10px 22px;
}

.pc-side-note strong {
  display: block;
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 31px;
  line-height: 1;
  margin: 9px 0 18px;
}

.pc-status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}

.pc-status {
  border: 1px solid var(--ink);
  border-radius: 999px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  padding: 7px 10px;
  text-transform: uppercase;
}

.pc-status.on { background: var(--acid); }
.pc-status.off { background: transparent; opacity: 0.55; }

.pc-card {
  background: var(--ink);
  color: var(--paper);
  display: grid;
  gap: 0;
  grid-template-columns: 0.72fr 1.28fr;
  margin: 18px 0 30px;
  min-height: 480px;
  overflow: hidden;
  position: relative;
}

.pc-card::after {
  border: 1px solid rgba(243, 239, 226, 0.35);
  content: "";
  inset: 14px;
  pointer-events: none;
  position: absolute;
}

.pc-card-art {
  align-items: flex-end;
  background:
    linear-gradient(145deg, transparent 38%, var(--rust) 38% 61%, transparent 61%),
    radial-gradient(circle at 30% 28%, var(--acid) 0 17%, transparent 17.5%),
    var(--paper);
  color: var(--ink);
  display: flex;
  padding: 46px 38px;
}

.pc-card-art span {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(62px, 9vw, 130px);
  line-height: 0.65;
}

.pc-card-copy {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 52px 48px;
}

.pc-card h2 {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(42px, 5vw, 74px);
  font-weight: 500;
  letter-spacing: -0.045em;
  line-height: 0.92;
  margin: 18px 0;
}

.pc-traits {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 24px;
}

.pc-traits span {
  border: 1px solid rgba(243, 239, 226, 0.42);
  border-radius: 999px;
  font-size: 12px;
  padding: 7px 10px;
}

.pc-meta {
  border-top: 1px solid rgba(243, 239, 226, 0.28);
  display: grid;
  font-size: 13px;
  gap: 18px;
  grid-template-columns: 1fr 1fr;
  line-height: 1.45;
  padding-top: 22px;
}

.pc-demo-title {
  border-top: 1px solid var(--ink);
  margin-top: 42px;
  padding: 32px 0 10px;
}

.pc-demo-title h3 {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(38px, 5vw, 64px);
  font-weight: 500;
  letter-spacing: -0.04em;
  margin: 8px 0;
}

.pc-library {
  border-top: 1px solid var(--ink);
  gap: 24px !important;
  margin-top: 56px !important;
  padding: 40px 0 20px;
}

.pc-library h3 {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(38px, 5vw, 64px);
  font-weight: 500;
  letter-spacing: -0.04em;
  margin: 8px 0 14px;
}

.pc-account, .pc-library-state {
  border: 1px solid var(--ink);
  min-height: 150px;
  padding: 22px;
}

.pc-account strong {
  display: block;
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 28px;
  margin: 8px 0;
}

.pc-dev-identity {
  background: var(--acid);
  color: var(--ink);
  display: inline-block;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  margin-top: 12px;
  padding: 6px 8px;
  text-transform: uppercase;
}

.pc-login button {
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  font-weight: 800 !important;
}

.pc-login-unavailable {
  border: 1px dashed var(--ink);
  margin-bottom: 8px;
  padding: 12px 14px;
}

.pc-create {
  border-top: 1px solid var(--ink);
  margin-top: 56px;
  padding-top: 40px;
}

.pc-create h3 {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(38px, 5vw, 64px);
  font-weight: 500;
  letter-spacing: -0.04em;
  margin: 8px 0 14px;
}

.pc-create-panel {
  background: rgba(243, 239, 226, 0.78) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  padding: 20px !important;
}

.pc-privacy-note {
  border-left: 4px solid var(--acid);
  margin: 14px 0;
  padding: 6px 0 6px 14px;
}

.pc-controls {
  background: rgba(243, 239, 226, 0.78) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  box-shadow: 10px 10px 0 var(--rust) !important;
  padding: 20px !important;
}

.pc-button {
  background: var(--ink) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  color: var(--paper) !important;
  font-weight: 800 !important;
  min-height: 48px;
}

@media (max-width: 760px) {
  .gradio-container { padding: 14px 14px 56px !important; }
  .pc-hero { grid-template-columns: 1fr; padding-top: 46px; }
  .pc-side-note { border-left: 0; border-top: 1px solid var(--ink); padding: 20px 0 0; }
  .pc-card { grid-template-columns: 1fr; }
  .pc-card-art { min-height: 250px; }
  .pc-card-copy { padding: 40px 30px 48px; }
  .pc-meta { grid-template-columns: 1fr; }
  .pc-library { flex-direction: column !important; }
}
"""


def _status_badges(settings: Settings) -> str:
    labels = {
        "hugging_face": "Hugging Face",
        "modal": "Modal",
        "elevenlabs": "ElevenLabs",
    }
    badges = []
    for provider, available in settings.providers.items():
        state = "on" if available else "off"
        suffix = "ready" if available else "not configured"
        badges.append(f'<span class="pc-status {state}">{labels[provider]} · {suffix}</span>')
    return "".join(badges)


def _landing_html(settings: Settings) -> str:
    return f"""
    <main>
      <nav class="pc-nav">
        <span class="pc-wordmark">Persona Capsule</span>
        <span class="pc-kicker">Build Small · 2026</span>
      </nav>
      <section class="pc-hero">
        <div>
          <span class="pc-kicker">Portable personality infrastructure</span>
          <h1>Your voice,<br><em>made tangible.</em></h1>
          <p class="pc-deck">
            Turn the patterns in how you communicate into a private, steerable,
            collectible capsule—then speak, fuse, battle, and share it.
          </p>
        </div>
        <aside class="pc-side-note">
          <span class="pc-label">Runtime principle</span>
          <strong>Steer live.<br>Store less.</strong>
          <p>
            Activation vectors are derived during inference and remain
            request-scoped. Your approved source material stays under your control.
          </p>
          <div class="pc-status-row">{_status_badges(settings)}</div>
        </aside>
      </section>
    </main>
    """


def _capsule_html() -> str:
    traits = "".join(f"<span>{escape(trait)}</span>" for trait in DEMO_CAPSULE.traits)
    return f"""
    <article class="pc-card">
      <div class="pc-card-art"><span>01</span></div>
      <div class="pc-card-copy">
        <div>
          <span class="pc-label">{escape(DEMO_CAPSULE.name)}</span>
          <h2>{escape(DEMO_CAPSULE.archetype)}</h2>
          <p>{escape(DEMO_CAPSULE.signal)}</p>
          <div class="pc-traits">{traits}</div>
        </div>
        <div class="pc-meta">
          <div><span class="pc-label">Cadence</span><br>{escape(DEMO_CAPSULE.cadence)}</div>
          <div>
            <span class="pc-label">Steering</span><br>
            {escape(DEMO_CAPSULE.steering_recipe)}
          </div>
        </div>
      </div>
    </article>
    """


def _account_html(principal: Principal | None) -> str:
    if principal is None:
        return """
        <div class="pc-account">
          <span class="pc-label">Creator access</span>
          <strong>Signed out</strong>
          <p>Sign in with Hugging Face to create and manage private capsules.</p>
        </div>
        """

    source = (
        '<span class="pc-dev-identity">Local development identity</span>'
        if principal.source == "local_development"
        else ""
    )
    return f"""
    <div class="pc-account">
      <span class="pc-label">Authenticated creator</span>
      <strong>@{escape(principal.username)}</strong>
      <p>Private capsules are scoped to this Hugging Face identity.</p>
      {source}
    </div>
    """


def _library_html(principal: Principal | None, capsule_library: CapsuleLibrary) -> str:
    if principal is None:
        return """
        <div class="pc-library-state">
          <span class="pc-label">Private library</span>
          <p>Locked. Public demos remain available, but creator operations require login.</p>
        </div>
        """

    records = capsule_library.list_capsules(principal)
    if not records:
        return """
        <div class="pc-library-state">
          <span class="pc-label">Private library · 0 capsules</span>
          <p>Your library is empty. Capsule creation opens in the next product slice.</p>
        </div>
        """

    items = "".join(
        f"<li><strong>{escape(record.name)}</strong> · {escape(record.status)}</li>"
        for record in records
    )
    return f"""
    <div class="pc-library-state">
      <span class="pc-label">Private library · {len(records)} capsules</span>
      <ul>{items}</ul>
    </div>
    """


def _capsule_choices(
    principal: Principal | None,
    capsule_library: CapsuleLibrary,
) -> list[tuple[str, str]]:
    if principal is None:
        return []
    return [
        (f"{record.name} · {record.status}", record.capsule_id)
        for record in capsule_library.list_capsules(principal)
    ]


def _draft_preview(draft: IngestionDraft) -> str:
    messages = "\n".join(
        f"{index + 1}. {escape(message.text)}" for index, message in enumerate(draft.messages)
    )
    redaction_counts: dict[str, int] = {}
    for redaction in draft.redactions:
        redaction_counts[redaction.kind] = redaction_counts.get(redaction.kind, 0) + 1
    summary = ", ".join(
        f"{kind.lower()}: {count}" for kind, count in sorted(redaction_counts.items())
    )
    if not summary:
        summary = "none detected"
    return (
        f"### Cleaned sample\n\nRedactions: **{summary}**\n\n"
        f"{messages}\n\n"
        "_Original pasted text remains only in this active draft and is discarded on approval._"
    )


def _pair_rows(draft: IngestionDraft) -> list[list[object]]:
    return [[True, pair.positive, pair.neutral] for pair in draft.proposed_pairs]


def _selected_pair_hashes(rows: object, draft: IngestionDraft) -> set[str]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    if not isinstance(rows, list):
        return set()
    selected: set[str] = set()
    for index, row in enumerate(rows):
        if index >= len(draft.proposed_pairs) or not isinstance(row, (list, tuple)):
            continue
        if row and bool(row[0]):
            selected.add(draft.proposed_pairs[index].pair_hash)
    return selected


def build_theme() -> gr.Theme:
    """Return the project theme at the Gradio app boundary."""

    return gr.themes.Base(
        primary_hue="orange",
        secondary_hue="lime",
        neutral_hue="stone",
        radius_size="none",
        font=["Avenir Next", "Gill Sans", "sans-serif"],
        font_mono=["SFMono-Regular", "Consolas", "monospace"],
    )


def build_demo(
    settings: Settings,
    identity_gateway: IdentityGateway,
    capsule_library: CapsuleLibrary,
    steering_service: CapsuleSteeringService,
    card_service: CapsuleCardService,
    publishing_service: PublishingService,
) -> gr.Blocks:
    """Build the public app shell and deterministic demo path."""

    def analyze_core(
        raw_input: str,
        speaker: str,
        consent: bool,
        principal: Principal | None,
    ) -> tuple[object, ...]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before creating a capsule.")
        try:
            draft = build_ingestion_draft(raw_input, speaker, consent)
        except IngestionError as exc:
            raise gr.Error(str(exc)) from exc
        dimensions = draft.profile.dimensions
        return (
            _draft_preview(draft),
            draft.profile.as_dict(),
            _pair_rows(draft),
            draft,
            dimensions.openness,
            dimensions.conscientiousness,
            dimensions.expressiveness,
            dimensions.agreeableness,
            dimensions.emotional_range,
            dimensions.directness,
            dimensions.formality,
            "Profile ready for review. Edit controls and choose the pairs to retain.",
        )

    def approve_core(
        capsule_name: str,
        draft: IngestionDraft | None,
        pair_rows: object,
        openness: float,
        conscientiousness: float,
        expressiveness: float,
        agreeableness: float,
        emotional_range: float,
        directness: float,
        formality: float,
        principal: Principal | None,
    ) -> tuple[str, str, str, str, None, CapsuleRecord, object]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before saving a capsule.")
        if draft is None:
            raise gr.Error("Analyze a message sample before approval.")
        name = capsule_name.strip()
        if not name:
            raise gr.Error("Give the capsule a name before approval.")
        try:
            approved = approve_draft(
                draft,
                _selected_pair_hashes(pair_rows, draft),
                {
                    "openness": openness,
                    "conscientiousness": conscientiousness,
                    "expressiveness": expressiveness,
                    "agreeableness": agreeableness,
                    "emotional_range": emotional_range,
                    "directness": directness,
                    "formality": formality,
                },
            )
        except IngestionError as exc:
            raise gr.Error(str(exc)) from exc
        record = capsule_library.save_capsule(
            principal,
            CapsuleRecord(
                capsule_id=uuid4().hex,
                owner_id=principal.user_id,
                name=name,
                status="profile_approved",
                style_profile=approved.profile,
                exemplar_pairs=approved.exemplar_pairs,
                source_fingerprint=approved.source_fingerprint,
            ),
        )
        return (
            (
                f"Approved **{escape(name)}** with "
                f"{len(approved.exemplar_pairs)} private steering pairs. "
                "The original pasted input and unselected messages were discarded."
            ),
            _library_html(principal, capsule_library),
            "",
            "",
            None,
            record,
            gr.Dropdown(
                choices=_capsule_choices(principal, capsule_library),
                value=record.capsule_id,
            ),
        )

    def steer_core(
        prompt: str,
        strength: float,
        capsule: CapsuleRecord | None,
        principal: Principal | None,
    ) -> tuple[str, str, object, str]:
        try:
            result = steering_service.compare(principal, capsule, prompt, strength)
        except (KeyError, PermissionError, SteeringError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        diagnostics = result["diagnostics"]
        warning = diagnostics.get("quality_warning")
        status = (
            f"Derived **{len(diagnostics['layers'])} live directions** from "
            f"**{diagnostics['exemplar_count']} approved pairs**. "
            f"Cache hit: **{diagnostics['cache_hit']}**. "
            f"Hooks active after request: **{diagnostics['hooks_active_after_request']}**."
        )
        if warning:
            status = f"**Quality warning:** {escape(str(warning))}\n\n{status}"
        return result["baseline"], result["steered"], diagnostics, status

    def generate_card_core(
        capsule_id: str,
        variation: str,
        seed: float,
        principal: Principal | None,
    ) -> tuple[str, str, str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before generating card art.")
        if not capsule_id:
            raise gr.Error("Approve or open a capsule before generating its card.")
        try:
            result = card_service.generate(
                principal,
                capsule_id,
                variation=variation,
                seed=int(seed),
            )
        except (KeyError, PermissionError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        provider_label = (
            "deterministic fallback" if result.used_fallback else "FLUX.2 Klein 4B on Modal"
        )
        status = (
            f"Generated **{escape(result.record.name)}** using **{provider_label}**. "
            f"Variation: **{result.prompt.variation}** · seed: **{result.prompt.seed}**. "
            "The approved profile was not modified."
        )
        return (
            str(result.interactive_path),
            str(result.social_path),
            status,
            result.record,
        )

    def publish_selection(
        include_summary: bool,
        include_descriptors: bool,
        include_dimensions: bool,
        include_card: bool,
    ) -> PublishSelection:
        return PublishSelection(
            include_summary=include_summary,
            include_descriptors=include_descriptors,
            include_dimensions=include_dimensions,
            include_card=include_card,
        )

    def preview_publish_core(
        capsule_id: str,
        include_summary: bool,
        include_descriptors: bool,
        include_dimensions: bool,
        include_card: bool,
        principal: Principal | None,
    ) -> tuple[object, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before previewing publication.")
        if not capsule_id:
            raise gr.Error("Choose a capsule to publish.")
        try:
            projection = publishing_service.preview(
                principal,
                capsule_id,
                publish_selection(
                    include_summary,
                    include_descriptors,
                    include_dimensions,
                    include_card,
                ),
            )
        except (KeyError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        return projection, "Review every field below, then explicitly confirm publication."

    def publish_core(
        capsule_id: str,
        include_summary: bool,
        include_descriptors: bool,
        include_dimensions: bool,
        include_card: bool,
        confirmed: bool,
        principal: Principal | None,
    ) -> tuple[str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before publishing.")
        if not capsule_id:
            raise gr.Error("Choose a capsule to publish.")
        try:
            record = publishing_service.publish(
                principal,
                capsule_id,
                publish_selection(
                    include_summary,
                    include_descriptors,
                    include_dimensions,
                    include_card,
                ),
                confirmed=confirmed,
            )
        except (KeyError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        public_url = publishing_service.public_url(record)
        x_url = publishing_service.x_share_url(record)
        return (
            f"Published at [{escape(public_url)}]({escape(public_url)}). "
            f"[Share this capsule on X]({escape(x_url)}).",
            record,
        )

    def unpublish_core(
        capsule_id: str,
        principal: Principal | None,
    ) -> tuple[str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before unpublishing.")
        if not capsule_id:
            raise gr.Error("Choose a capsule to unpublish.")
        try:
            record = publishing_service.unpublish(principal, capsule_id)
        except KeyError as exc:
            raise gr.Error(str(exc)) from exc
        return (
            "Public page disabled. The private capsule and generated card remain available.",
            record,
        )

    def open_capsule_core(
        capsule_id: str,
        principal: Principal | None,
    ) -> tuple[CapsuleRecord, object, list[list[object]], str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before opening a capsule.")
        if not capsule_id:
            raise gr.Error("Choose a capsule from your library.")
        record = capsule_library.get_capsule(principal, capsule_id)
        if record.style_profile is None:
            raise gr.Error("This capsule does not have an approved profile.")
        return (
            record,
            record.style_profile.as_dict(include_private_evidence=True),
            [[True, pair.positive, pair.neutral] for pair in record.exemplar_pairs],
            f"Opened **{escape(record.name)}** from durable private storage.",
        )

    def export_capsule_core(
        capsule_id: str,
        include_private_exemplars: bool,
        principal: Principal | None,
    ) -> tuple[str, str, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before exporting a capsule.")
        if not capsule_id:
            raise gr.Error("Choose a capsule to export.")
        record = capsule_library.get_capsule(principal, capsule_id)
        bundle = build_capsule_export(
            record,
            include_private_exemplars=include_private_exemplars,
        )
        export_dir = (
            Path(settings.capsule_data_dir)
            / "exports"
            / sha256(principal.user_id.encode()).hexdigest()[:24]
            / record.capsule_id
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        persona_path = export_dir / bundle.persona_filename
        manifest_path = export_dir / bundle.manifest_filename
        persona_path.write_bytes(bundle.persona_bytes)
        manifest_path.write_bytes(bundle.manifest_bytes)
        persona_path.chmod(0o600)
        manifest_path.chmod(0o600)
        return (
            str(persona_path),
            str(manifest_path),
            (
                "Export ready. Private exemplar pairs were included."
                if include_private_exemplars
                else "Export ready. Private exemplar pairs were excluded."
            ),
        )

    def delete_capsule_core(
        capsule_id: str,
        confirmed: bool,
        principal: Principal | None,
    ) -> tuple[str, str, object, None]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before deleting a capsule.")
        if not capsule_id:
            raise gr.Error("Choose a capsule to delete.")
        if not confirmed:
            raise gr.Error("Confirm permanent deletion first.")
        deleted = capsule_library.delete_capsule(principal, capsule_id)
        cache_status = "Warm steering cache invalidated."
        try:
            steering_service.invalidate(principal, capsule_id)
        except Exception:
            cache_status = "Remote cache expiry will finish cleanup."
        choices = _capsule_choices(principal, capsule_library)
        status = (
            f"Capsule deleted and local artifacts removed. {cache_status}"
            if deleted
            else f"Capsule was already deleted. {cache_status}"
        )
        return (
            status,
            _library_html(principal, capsule_library),
            gr.Dropdown(choices=choices, value=None),
            None,
        )

    with gr.Blocks(title="Persona Capsule — Your voice, made tangible") as demo:
        gr.HTML(_landing_html(settings))
        gr.HTML(_capsule_html())
        with gr.Row(elem_classes=["pc-library"]):
            with gr.Column(scale=7):
                gr.HTML(
                    """
                    <span class="pc-kicker">Identity-bound by design</span>
                    <h3>Your private library.</h3>
                    <p>
                      Public personality cards can travel. Source material, drafts,
                      and creator controls stay attached to your Hugging Face identity.
                    </p>
                    """
                )
            with gr.Column(scale=13):
                if settings.oauth_ui_available:
                    gr.LoginButton("Sign in with Hugging Face", elem_classes=["pc-login"])
                else:
                    gr.HTML(
                        """
                        <div class="pc-login-unavailable">
                          Hugging Face login activates in the Space. For local OAuth
                          testing, configure <code>HF_TOKEN</code>.
                        </div>
                        """
                    )
                account = gr.HTML()
                library = gr.HTML()
        with gr.Group(elem_classes=["pc-create-panel"]):
            gr.Markdown("#### Open, export, or delete a private capsule")
            with gr.Row():
                capsule_selector = gr.Dropdown(
                    label="Private capsule",
                    choices=[],
                )
                open_capsule = gr.Button("Open capsule")
            lifecycle_status = gr.Markdown()
            reopened_profile = gr.JSON(label="Canonical private profile")
            reopened_pairs = gr.Dataframe(
                headers=["Approved", "Style exemplar", "Neutral contrast"],
                datatype=["bool", "str", "str"],
                interactive=False,
                label="Approved private steering pairs",
            )
            include_export_exemplars = gr.Checkbox(
                label="Include approved private exemplar pairs in this export",
                value=False,
            )
            with gr.Row():
                export_capsule = gr.Button("Export .persona and manifest")
                delete_confirmation = gr.Checkbox(
                    label="I understand this permanently deletes the capsule.",
                    value=False,
                )
                delete_capsule = gr.Button("Delete capsule")
            with gr.Row():
                persona_export = gr.File(label=".persona export")
                manifest_export = gr.File(label="Compatibility manifest")
        gr.HTML(
            """
            <section class="pc-create">
              <span class="pc-kicker">Quick Capsule · private by default</span>
              <h3>Approve the signal, not the archive.</h3>
              <p>
                Paste your own messages, review redactions and the inferred profile,
                then choose the small set of examples retained for live steering.
              </p>
              <div class="pc-privacy-note">
                This describes communication style, not mental health, identity,
                intelligence, or a clinical personality diagnosis.
              </div>
            </section>
            """
        )
        draft_state = gr.State(value=None)
        approved_capsule_state = gr.State(value=None)
        with gr.Group(elem_classes=["pc-create-panel"]):
            with gr.Row():
                capsule_name = gr.Textbox(
                    label="Capsule name",
                    placeholder="e.g. Clear Signal",
                )
                speaker = gr.Textbox(
                    label="Your speaker label",
                    value="You",
                    info='For "Name: message" exports, enter the matching name.',
                )
            raw_messages = gr.Textbox(
                label="Your messages",
                lines=10,
                placeholder=(
                    "You: Thanks — I see the tradeoff. Let’s test the smallest version first.\n"
                    "You: I’m not convinced yet; what would change your mind?\n"
                    "…paste at least 8 varied messages, ideally around 20."
                ),
            )
            consent = gr.Checkbox(
                label="I own these messages or have permission to process them.",
                value=False,
            )
            analyze = gr.Button("Analyze private sample", elem_classes=["pc-button"])
            creation_status = gr.Markdown()
            cleaned_preview = gr.Markdown()
            profile_json = gr.JSON(label="Editable profile evidence")
            pair_table = gr.Dataframe(
                headers=["Retain", "Style exemplar", "Neutral contrast"],
                datatype=["bool", "str", "str"],
                interactive=True,
                label="Private steering pairs",
            )
            gr.Markdown("#### OCEAN-inspired style controls")
            with gr.Row():
                openness = gr.Slider(0, 100, value=50, label="Openness")
                conscientiousness = gr.Slider(0, 100, value=50, label="Conscientiousness")
                expressiveness = gr.Slider(0, 100, value=50, label="Expressiveness")
            with gr.Row():
                agreeableness = gr.Slider(0, 100, value=50, label="Agreeableness")
                emotional_range = gr.Slider(0, 100, value=50, label="Emotional range")
                directness = gr.Slider(0, 100, value=50, label="Directness")
                formality = gr.Slider(0, 100, value=50, label="Formality")
            approve = gr.Button(
                "Approve profile and retained pairs",
                elem_classes=["pc-button"],
            )
        gr.HTML(
            """
            <section class="pc-demo-title">
              <span class="pc-kicker">Collectible identity · FLUX.2 Klein</span>
              <h3>Turn the profile into an object.</h3>
              <p>
                The visual prompt uses only approved profile dimensions and mapped
                public descriptors. Private messages and evidence never enter it.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["pc-controls"]):
            with gr.Row():
                card_variation = gr.Dropdown(
                    choices=["signal", "archive", "kinetic"],
                    value="signal",
                    label="Controlled variation",
                )
                card_seed = gr.Number(
                    value=7613,
                    precision=0,
                    label="Visual seed",
                )
            generate_card = gr.Button(
                "Generate collectible card",
                elem_classes=["pc-button"],
            )
            card_status = gr.Markdown()
            with gr.Row():
                interactive_card = gr.Image(
                    label="Interactive card · 768×1024",
                    type="filepath",
                )
                social_card = gr.Image(
                    label="Social preview · 1200×628",
                    type="filepath",
                )
        gr.HTML(
            """
            <section class="pc-demo-title">
              <span class="pc-kicker">Public projection · private source</span>
              <h3>Choose exactly what travels.</h3>
              <p>
                Preview selected fields before publishing an opaque stable URL.
                Private exemplars, evidence, owner IDs, and provider references
                are never part of the public projection.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["pc-controls"]):
            with gr.Row():
                publish_summary = gr.Checkbox(
                    label="Public summary",
                    value=True,
                )
                publish_descriptors = gr.Checkbox(
                    label="Public descriptors",
                    value=True,
                )
                publish_dimensions = gr.Checkbox(
                    label="Public style dimensions",
                    value=False,
                )
                publish_card = gr.Checkbox(
                    label="Public social card",
                    value=True,
                )
            preview_publish = gr.Button("Preview public projection")
            public_preview = gr.JSON(label="Fields visible without login")
            publish_confirmation = gr.Checkbox(
                label="I reviewed this projection and want to make it public.",
                value=False,
            )
            with gr.Row():
                publish_capsule = gr.Button(
                    "Publish stable capsule URL",
                    elem_classes=["pc-button"],
                )
                unpublish_capsule = gr.Button("Unpublish capsule")
            publish_status = gr.Markdown()
        gr.HTML(
            """
            <section class="pc-demo-title">
              <span class="pc-kicker">Live inference · MiniCPM4.1-8B on Modal</span>
              <h3>See the steering vector work.</h3>
              <p>
                Compare one deterministic baseline against a response steered from
                the approved capsule pairs. Directions are derived inside the request,
                applied at five decoder layers, and then removed.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["pc-controls"]):
            live_prompt = gr.Textbox(
                label="Prompt",
                placeholder="Explain why a small team should test the risky assumption first.",
                lines=3,
            )
            live_strength = gr.Slider(
                -1.5,
                1.5,
                value=0.85,
                step=0.05,
                label="Inference-time steering strength",
                info="Values above ±1.1 may reduce coherence.",
            )
            live_run = gr.Button("Compare baseline vs live steering", elem_classes=["pc-button"])
            live_status = gr.Markdown()
            with gr.Row():
                baseline_output = gr.Textbox(label="MiniCPM baseline", lines=9)
                steered_output = gr.Textbox(label="Live-steered response", lines=9)
            vector_diagnostics = gr.JSON(
                label="Request-scoped vector diagnostics",
            )
        gr.HTML(
            """
            <section class="pc-demo-title">
              <span class="pc-kicker">Offline proof · no provider call</span>
              <h3>Try the capsule’s shape.</h3>
              <p>
                This deterministic preview proves the product path while the live
                MiniCPM steering runtime is built.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["pc-controls"]):
            prompt = gr.Textbox(
                label="Situation",
                placeholder="Explain a difficult technical decision to a teammate…",
                lines=3,
            )
            intensity = gr.Radio(
                choices=["Subtle", "Balanced", "Expressive"],
                value="Balanced",
                label="Capsule intensity",
            )
            run = gr.Button("Respond as Signal / No. 01", elem_classes=["pc-button"])
            output = gr.Markdown(label="Capsule response")
            run.click(demo_reply, inputs=[prompt, intensity], outputs=output)

        analysis_outputs = [
            cleaned_preview,
            profile_json,
            pair_table,
            draft_state,
            openness,
            conscientiousness,
            expressiveness,
            agreeableness,
            emotional_range,
            directness,
            formality,
            creation_status,
        ]
        approval_inputs = [
            capsule_name,
            draft_state,
            pair_table,
            openness,
            conscientiousness,
            expressiveness,
            agreeableness,
            emotional_range,
            directness,
            formality,
        ]
        approval_outputs = [
            creation_status,
            library,
            raw_messages,
            cleaned_preview,
            draft_state,
            approved_capsule_state,
            capsule_selector,
        ]

        if settings.oauth_ui_available:

            def load_private_library(
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, object]:
                principal = identity_gateway.resolve(profile)
                return (
                    _account_html(principal),
                    _library_html(principal, capsule_library),
                    gr.Dropdown(
                        choices=_capsule_choices(principal, capsule_library),
                        value=None,
                    ),
                )

            demo.load(
                load_private_library,
                inputs=None,
                outputs=[account, library, capsule_selector],
            )

            def analyze_with_oauth(
                raw_input: str,
                speaker_label: str,
                has_consent: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, ...]:
                return analyze_core(
                    raw_input,
                    speaker_label,
                    has_consent,
                    identity_gateway.resolve(profile),
                )

            def approve_with_oauth(
                name: str,
                draft: IngestionDraft | None,
                rows: object,
                open_value: float,
                conscientious_value: float,
                expressive_value: float,
                agreeable_value: float,
                emotional_value: float,
                direct_value: float,
                formal_value: float,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, str, str, None, CapsuleRecord, object]:
                return approve_core(
                    name,
                    draft,
                    rows,
                    open_value,
                    conscientious_value,
                    expressive_value,
                    agreeable_value,
                    emotional_value,
                    direct_value,
                    formal_value,
                    identity_gateway.resolve(profile),
                )

            analyze.click(
                analyze_with_oauth,
                inputs=[raw_messages, speaker, consent],
                outputs=analysis_outputs,
            )
            approve.click(
                approve_with_oauth,
                inputs=approval_inputs,
                outputs=approval_outputs,
            )

            def steer_with_oauth(
                prompt_value: str,
                strength_value: float,
                capsule: CapsuleRecord | None,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, object, str]:
                return steer_core(
                    prompt_value,
                    strength_value,
                    capsule,
                    identity_gateway.resolve(profile),
                )

            live_run.click(
                steer_with_oauth,
                inputs=[live_prompt, live_strength, approved_capsule_state],
                outputs=[
                    baseline_output,
                    steered_output,
                    vector_diagnostics,
                    live_status,
                ],
            )

            def generate_card_with_oauth(
                capsule_id: str,
                variation: str,
                seed: float,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, str, CapsuleRecord]:
                return generate_card_core(
                    capsule_id,
                    variation,
                    seed,
                    identity_gateway.resolve(profile),
                )

            generate_card.click(
                generate_card_with_oauth,
                inputs=[capsule_selector, card_variation, card_seed],
                outputs=[
                    interactive_card,
                    social_card,
                    card_status,
                    approved_capsule_state,
                ],
            )

            def preview_publish_with_oauth(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, str]:
                return preview_publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    identity_gateway.resolve(profile),
                )

            def publish_with_oauth(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                confirmed: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, CapsuleRecord]:
                return publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    confirmed,
                    identity_gateway.resolve(profile),
                )

            def unpublish_with_oauth(
                capsule_id: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, CapsuleRecord]:
                return unpublish_core(
                    capsule_id,
                    identity_gateway.resolve(profile),
                )

            publish_inputs = [
                capsule_selector,
                publish_summary,
                publish_descriptors,
                publish_dimensions,
                publish_card,
            ]
            preview_publish.click(
                preview_publish_with_oauth,
                inputs=publish_inputs,
                outputs=[public_preview, publish_status],
            )
            publish_capsule.click(
                publish_with_oauth,
                inputs=publish_inputs + [publish_confirmation],
                outputs=[publish_status, approved_capsule_state],
            )
            unpublish_capsule.click(
                unpublish_with_oauth,
                inputs=[capsule_selector],
                outputs=[publish_status, approved_capsule_state],
            )

            def principal_from_oauth(
                profile: gr.OAuthProfile | None,
            ) -> Principal | None:
                return identity_gateway.resolve(profile)

            def open_with_oauth(
                capsule_id: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[CapsuleRecord, object, list[list[object]], str]:
                return open_capsule_core(capsule_id, principal_from_oauth(profile))

            def export_with_oauth(
                capsule_id: str,
                include_private: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, str]:
                return export_capsule_core(
                    capsule_id,
                    include_private,
                    principal_from_oauth(profile),
                )

            def delete_with_oauth(
                capsule_id: str,
                confirmed: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, object, None]:
                return delete_capsule_core(
                    capsule_id,
                    confirmed,
                    principal_from_oauth(profile),
                )

            open_capsule.click(
                open_with_oauth,
                inputs=[capsule_selector],
                outputs=[
                    approved_capsule_state,
                    reopened_profile,
                    reopened_pairs,
                    lifecycle_status,
                ],
            )
            export_capsule.click(
                export_with_oauth,
                inputs=[capsule_selector, include_export_exemplars],
                outputs=[persona_export, manifest_export, lifecycle_status],
            )
            delete_capsule.click(
                delete_with_oauth,
                inputs=[capsule_selector, delete_confirmation],
                outputs=[
                    lifecycle_status,
                    library,
                    capsule_selector,
                    approved_capsule_state,
                ],
            )
        else:

            def load_local_library() -> tuple[str, str, object]:
                principal = identity_gateway.resolve_local()
                return (
                    _account_html(principal),
                    _library_html(principal, capsule_library),
                    gr.Dropdown(
                        choices=_capsule_choices(principal, capsule_library),
                        value=None,
                    ),
                )

            demo.load(
                load_local_library,
                inputs=None,
                outputs=[account, library, capsule_selector],
            )

            def analyze_locally(
                raw_input: str,
                speaker_label: str,
                has_consent: bool,
            ) -> tuple[object, ...]:
                return analyze_core(
                    raw_input,
                    speaker_label,
                    has_consent,
                    identity_gateway.resolve_local(),
                )

            def approve_locally(
                name: str,
                draft: IngestionDraft | None,
                rows: object,
                open_value: float,
                conscientious_value: float,
                expressive_value: float,
                agreeable_value: float,
                emotional_value: float,
                direct_value: float,
                formal_value: float,
            ) -> tuple[str, str, str, str, None, CapsuleRecord, object]:
                return approve_core(
                    name,
                    draft,
                    rows,
                    open_value,
                    conscientious_value,
                    expressive_value,
                    agreeable_value,
                    emotional_value,
                    direct_value,
                    formal_value,
                    identity_gateway.resolve_local(),
                )

            analyze.click(
                analyze_locally,
                inputs=[raw_messages, speaker, consent],
                outputs=analysis_outputs,
            )
            approve.click(
                approve_locally,
                inputs=approval_inputs,
                outputs=approval_outputs,
            )

            def steer_locally(
                prompt_value: str,
                strength_value: float,
                capsule: CapsuleRecord | None,
            ) -> tuple[str, str, object, str]:
                return steer_core(
                    prompt_value,
                    strength_value,
                    capsule,
                    identity_gateway.resolve_local(),
                )

            live_run.click(
                steer_locally,
                inputs=[live_prompt, live_strength, approved_capsule_state],
                outputs=[
                    baseline_output,
                    steered_output,
                    vector_diagnostics,
                    live_status,
                ],
            )

            def generate_card_locally(
                capsule_id: str,
                variation: str,
                seed: float,
            ) -> tuple[str, str, str, CapsuleRecord]:
                return generate_card_core(
                    capsule_id,
                    variation,
                    seed,
                    identity_gateway.resolve_local(),
                )

            generate_card.click(
                generate_card_locally,
                inputs=[capsule_selector, card_variation, card_seed],
                outputs=[
                    interactive_card,
                    social_card,
                    card_status,
                    approved_capsule_state,
                ],
            )

            def preview_publish_locally(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
            ) -> tuple[object, str]:
                return preview_publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    identity_gateway.resolve_local(),
                )

            def publish_locally(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                confirmed: bool,
            ) -> tuple[str, CapsuleRecord]:
                return publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    confirmed,
                    identity_gateway.resolve_local(),
                )

            def unpublish_locally(
                capsule_id: str,
            ) -> tuple[str, CapsuleRecord]:
                return unpublish_core(
                    capsule_id,
                    identity_gateway.resolve_local(),
                )

            local_publish_inputs = [
                capsule_selector,
                publish_summary,
                publish_descriptors,
                publish_dimensions,
                publish_card,
            ]
            preview_publish.click(
                preview_publish_locally,
                inputs=local_publish_inputs,
                outputs=[public_preview, publish_status],
            )
            publish_capsule.click(
                publish_locally,
                inputs=local_publish_inputs + [publish_confirmation],
                outputs=[publish_status, approved_capsule_state],
            )
            unpublish_capsule.click(
                unpublish_locally,
                inputs=[capsule_selector],
                outputs=[publish_status, approved_capsule_state],
            )

            def open_locally(
                capsule_id: str,
            ) -> tuple[CapsuleRecord, object, list[list[object]], str]:
                return open_capsule_core(
                    capsule_id,
                    identity_gateway.resolve_local(),
                )

            def export_locally(
                capsule_id: str,
                include_private: bool,
            ) -> tuple[str, str, str]:
                return export_capsule_core(
                    capsule_id,
                    include_private,
                    identity_gateway.resolve_local(),
                )

            def delete_locally(
                capsule_id: str,
                confirmed: bool,
            ) -> tuple[str, str, object, None]:
                return delete_capsule_core(
                    capsule_id,
                    confirmed,
                    identity_gateway.resolve_local(),
                )

            open_capsule.click(
                open_locally,
                inputs=[capsule_selector],
                outputs=[
                    approved_capsule_state,
                    reopened_profile,
                    reopened_pairs,
                    lifecycle_status,
                ],
            )
            export_capsule.click(
                export_locally,
                inputs=[capsule_selector, include_export_exemplars],
                outputs=[persona_export, manifest_export, lifecycle_status],
            )
            delete_capsule.click(
                delete_locally,
                inputs=[capsule_selector, delete_confirmation],
                outputs=[
                    lifecycle_status,
                    library,
                    capsule_selector,
                    approved_capsule_state,
                ],
            )

    return demo
