"""Custom Gradio interface for Persona Capsule."""

from dataclasses import replace
from hashlib import sha256
from html import escape
from pathlib import Path
from uuid import uuid4

import gradio as gr

from persona_capsule.battle import CapsuleBattleService
from persona_capsule.card import CapsuleCardService
from persona_capsule.config import Settings
from persona_capsule.deep_training import DeepCapsuleService, estimate_deep_training
from persona_capsule.demo import DEMO_CAPSULE, demo_reply
from persona_capsule.export import build_capsule_export
from persona_capsule.fusion import CapsuleFusionService
from persona_capsule.identity import IdentityGateway, Principal
from persona_capsule.ingestion import (
    IngestionDraft,
    IngestionError,
    MessageRecord,
    approve_draft,
    build_ingestion_draft,
    infer_style_profile,
)
from persona_capsule.library import CapsuleLibrary
from persona_capsule.operations import (
    FeatureDisabledError,
    OperationsGuard,
    QuotaExceededError,
)
from persona_capsule.publishing import PublishingService, PublishSelection
from persona_capsule.repository import CapsulePublicProjection, CapsuleRecord
from persona_capsule.steering import SteeringError
from persona_capsule.steering_service import CapsuleSteeringService
from persona_capsule.voice import CapsuleVoiceService, VoiceError

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
  margin: 0 auto !important;
  max-width: 1240px !important;
  padding: 22px 24px 72px !important;
  width: min(100%, 1240px) !important;
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

.pc-hero {
  align-items: end;
  padding: 34px 0 22px;
}

.pc-hero h1 {
  font-size: clamp(44px, 6vw, 76px);
  margin: 12px 0 16px;
}

.pc-hero-actions {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 20px;
}

.pc-start-pill {
  background: var(--ink);
  color: var(--paper);
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  padding: 11px 14px;
  text-transform: uppercase;
}

.pc-start-note {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 17px;
  font-style: italic;
}

.pc-route {
  display: grid;
  gap: 1px;
  grid-template-columns: repeat(3, 1fr);
  margin: 8px 0 28px;
  overflow: hidden;
}

.pc-route-card {
  background: rgba(243, 239, 226, 0.86);
  border: 1px solid var(--ink);
  color: var(--ink);
  min-height: 116px;
  padding: 15px 18px;
  position: relative;
}

.pc-route-card:nth-child(2) {
  background: var(--acid);
}

.pc-route-card:nth-child(3) {
  background: var(--rust);
  color: var(--paper);
}

.pc-route-number {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 27px;
  line-height: 1;
}

.pc-route-card strong {
  display: block;
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 21px;
  margin: 8px 0 4px;
}

.pc-route-card p {
  font-size: 13px;
  line-height: 1.45;
  margin: 0;
}

.pc-workspace {
  align-items: stretch;
  border-top: 1px solid var(--ink);
  gap: 18px !important;
  margin: 0 12px 22px !important;
  width: calc(100% - 24px) !important;
  padding-top: 22px;
}

.pc-section-intro h3 {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: clamp(34px, 4vw, 52px);
  font-weight: 500;
  letter-spacing: -0.04em;
  line-height: 0.95;
  margin: 8px 0 14px;
}

.pc-workspace-intro {
  display: flex !important;
  flex-direction: column;
  gap: 10px;
  justify-content: space-between;
}

.pc-workspace-note {
  border-left: 4px solid var(--rust);
  font-family: "Iowan Old Style", Baskerville, Georgia, serif;
  font-size: 18px;
  line-height: 1.35;
  margin-bottom: 4px;
  padding: 4px 0 4px 13px;
}

.pc-account {
  min-height: 0;
  padding: 15px;
}

.pc-account strong {
  font-size: 23px;
  margin: 4px 0;
}

.pc-account p {
  font-size: 12px;
  margin: 4px 0;
}

.pc-active-capsule {
  background: var(--ink) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  color: var(--paper) !important;
  padding: 15px !important;
}

.pc-active-capsule label,
.pc-active-capsule .prose,
.pc-active-capsule p {
  color: var(--paper) !important;
}

.pc-active-capsule button {
  background: var(--acid) !important;
  border: 1px solid var(--acid) !important;
  color: var(--ink) !important;
  font-weight: 900 !important;
}

.pc-library-state {
  background: rgba(243, 239, 226, 0.72);
  display: none;
  min-height: 0;
  padding: 13px;
}

.pc-library-state ul {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  list-style: none;
  margin: 12px 0 0;
  padding: 0;
}

.pc-library-state li {
  border: 1px solid var(--ink);
  font-size: 12px;
  padding: 7px 9px;
}

.pc-journey-tabs {
  margin: 18px 12px 0;
  width: calc(100% - 24px);
}

.pc-journey-tabs > .tab-nav {
  background: transparent !important;
  border-bottom: 1px solid var(--ink) !important;
  gap: 4px;
  padding: 0 !important;
  position: sticky;
  top: 0;
  z-index: 20;
}

.pc-journey-tabs > .tab-nav button {
  background: var(--paper) !important;
  border: 1px solid var(--ink) !important;
  border-bottom: 0 !important;
  border-radius: 0 !important;
  color: var(--ink) !important;
  font-size: 12px !important;
  font-weight: 900 !important;
  letter-spacing: 0.08em;
  min-height: 48px;
  padding: 12px 18px !important;
  text-transform: uppercase;
}

.pc-journey-tabs > .tab-nav button.selected {
  background: var(--ink) !important;
  color: var(--paper) !important;
}

.pc-tab-shell {
  animation: pc-rise 420ms ease both;
  padding-top: 30px;
}

.pc-section-intro {
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 0.75fr) minmax(320px, 1.25fr);
  margin-bottom: 22px;
}

.pc-section-intro p {
  font-size: 16px;
  line-height: 1.55;
  margin: 0;
  max-width: 680px;
}

.pc-step-chip {
  align-self: start;
  background: var(--acid);
  border: 1px solid var(--ink);
  display: inline-block;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.12em;
  padding: 7px 9px;
  text-transform: uppercase;
}

.pc-primary-panel {
  background: rgba(243, 239, 226, 0.9) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  box-shadow: 8px 8px 0 var(--rust) !important;
  padding: 22px !important;
}

.pc-result-panel {
  border-left: 5px solid var(--acid);
  margin-top: 14px;
  padding-left: 14px;
}

.pc-advanced {
  border: 1px solid var(--ink) !important;
  border-radius: 0 !important;
  margin-top: 16px !important;
}

.pc-advanced > button,
.pc-advanced > .label-wrap {
  font-family: "Iowan Old Style", Baskerville, Georgia, serif !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}

.pc-danger-zone {
  border-color: var(--rust) !important;
}

.pc-help {
  background: rgba(216, 242, 74, 0.22);
  border-left: 4px solid var(--acid);
  font-size: 13px;
  line-height: 1.5;
  margin: 12px 0 18px;
  padding: 12px 14px;
}

@keyframes pc-rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 760px) {
  .gradio-container { padding: 14px 14px 56px !important; }
  .pc-nav { align-items: flex-start; }
  .pc-nav .pc-kicker { max-width: 150px; text-align: right; }
  .pc-hero { grid-template-columns: 1fr; padding: 28px 0 20px; }
  .pc-hero h1 { font-size: 48px; }
  .pc-side-note { display: none; }
  .pc-card { grid-template-columns: 1fr; }
  .pc-card-art { min-height: 250px; }
  .pc-card-copy { padding: 40px 30px 48px; }
  .pc-meta { grid-template-columns: 1fr; }
  .pc-library { flex-direction: column !important; }
  .pc-route { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .pc-route-card { min-height: 96px; padding: 12px 10px; }
  .pc-route-card p { display: none; }
  .pc-route-card strong { font-size: 16px; margin-top: 10px; }
  .pc-route-number { font-size: 21px; }
  .pc-section-intro { grid-template-columns: 1fr; }
  .pc-workspace { flex-direction: column !important; }
  .pc-workspace {
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
  }
  .pc-workspace-intro { display: none !important; }
  .pc-journey-tabs {
    margin-left: 0;
    margin-right: 0;
    width: 100%;
  }
  .pc-active-capsule { padding: 12px !important; }
  .pc-journey-tabs > .tab-nav {
    overflow-x: auto;
    position: static;
  }
  .pc-journey-tabs > .tab-nav button {
    flex: 0 0 auto;
    min-width: 150px;
  }
}
"""


def _status_badges(settings: Settings) -> str:
    labels = {
        "hugging_face": "Hugging Face",
        "modal": "Modal",
        "voxcpm2": "VoxCPM2",
    }
    badges = []
    for provider, available in settings.providers.items():
        state = "on" if available else "off"
        suffix = "configured" if available else "not configured"
        badges.append(f'<span class="pc-status {state}">{labels[provider]} · {suffix}</span>')
    return "".join(badges)


def _landing_html(settings: Settings) -> str:
    return f"""
    <main>
      <nav class="pc-nav">
        <span class="pc-wordmark">Persona Capsule</span>
        <span class="pc-kicker">Build Small · Personality Studio</span>
      </nav>
      <section class="pc-hero">
        <div>
          <span class="pc-kicker">A private, steerable version of your communication style</span>
          <h1>Make your voice<br><em>portable.</em></h1>
          <p class="pc-deck">
            Bring a few messages. Persona Capsule learns the patterns in how you
            communicate, lets you test them live, and turns them into something
            you can see, hear, and share.
          </p>
          <div class="pc-hero-actions">
            <span class="pc-start-pill">Start with step 1 below</span>
            <span class="pc-start-note">Your source messages stay private.</span>
          </div>
        </div>
        <aside class="pc-side-note">
          <span class="pc-label">What makes it different</span>
          <strong>It changes the model while it writes.</strong>
          <p>
            The personality direction is calculated at inference time, applied for
            one response, then removed. No permanent steering tensor is stored.
          </p>
          <div class="pc-status-row">{_status_badges(settings)}</div>
        </aside>
      </section>
      <section class="pc-route" aria-label="Product journey">
        <article class="pc-route-card">
          <span class="pc-route-number">01</span>
          <strong>Create</strong>
          <p>Paste messages, review the profile, and approve a few examples.</p>
        </article>
        <article class="pc-route-card">
          <span class="pc-route-number">02</span>
          <strong>Test</strong>
          <p>Compare MiniCPM before and after live activation steering.</p>
        </article>
        <article class="pc-route-card">
          <span class="pc-route-number">03</span>
          <strong>Bring it to life</strong>
          <p>Generate the card, add an authorized voice, and share on X.</p>
        </article>
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


def _steerable_capsule_choices(
    principal: Principal | None,
    capsule_library: CapsuleLibrary,
) -> list[tuple[str, str]]:
    if principal is None:
        return []
    return [
        (f"{record.name} · {record.status}", record.capsule_id)
        for record in capsule_library.list_capsules(principal)
        if record.style_profile is not None and record.exemplar_pairs
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


def _recover_ingestion_draft(
    draft: IngestionDraft | None,
    raw_input: str,
    speaker: str,
    consent: bool,
) -> IngestionDraft:
    """Rebuild deterministic analysis when browser session state is unavailable."""

    if draft is not None:
        return draft
    return build_ingestion_draft(raw_input, speaker, consent)


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
    voice_service: CapsuleVoiceService,
    fusion_service: CapsuleFusionService,
    battle_service: CapsuleBattleService,
    deep_service: DeepCapsuleService,
    operations: OperationsGuard,
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
        raw_input: str,
        speaker_label: str,
        has_consent: bool,
        pair_rows: object,
        openness: float,
        conscientiousness: float,
        expressiveness: float,
        agreeableness: float,
        emotional_range: float,
        directness: float,
        formality: float,
        principal: Principal | None,
    ) -> tuple[str, str, str, str, str, None, CapsuleRecord, object, object]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before saving a capsule.")
        try:
            resolved_draft = _recover_ingestion_draft(
                draft,
                raw_input,
                speaker_label,
                has_consent,
            )
        except IngestionError as exc:
            raise gr.Error(str(exc)) from exc
        name = capsule_name.strip()
        if not name:
            raise gr.Error("Give the capsule a name before approval.")
        try:
            operations.require(principal.user_id, "creation")
        except (FeatureDisabledError, QuotaExceededError) as exc:
            raise gr.Error(str(exc)) from exc
        try:
            approved = approve_draft(
                resolved_draft,
                _selected_pair_hashes(pair_rows, resolved_draft),
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
        gr.Info(
            f"{name} is ready. Opening Step 2 so you can test its live steering.",
            duration=8,
            title="Capsule created",
        )
        return (
            (
                f"Approved **{escape(name)}** with "
                f"{len(approved.exemplar_pairs)} private steering pairs. "
                "The original pasted input and unselected messages were discarded."
            ),
            _library_html(principal, capsule_library),
            (
                f"Active capsule: **{escape(name)}** · "
                f"{len(approved.exemplar_pairs)} approved steering pairs."
            ),
            "",
            "",
            None,
            record,
            gr.Dropdown(
                choices=_capsule_choices(principal, capsule_library),
                value=record.capsule_id,
            ),
            gr.Tabs(selected="test"),
        )

    def steer_core(
        prompt: str,
        strength: float,
        capsule: CapsuleRecord | None,
        principal: Principal | None,
    ) -> tuple[str, str, object, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before running live steering.")
        try:
            operations.require(principal.user_id, "steering")
            result = steering_service.compare(principal, capsule, prompt, strength)
        except (
            FeatureDisabledError,
            KeyError,
            PermissionError,
            QuotaExceededError,
            SteeringError,
            ValueError,
        ) as exc:
            raise gr.Error(str(exc)) from exc
        diagnostics = result["diagnostics"]
        warning = diagnostics.get("quality_warning")
        calibration_norms = [
            float(layer.get("calibration_norm", 1.0)) for layer in diagnostics["layers"]
        ]
        calibration_summary = ""
        if calibration_norms:
            calibration_summary = (
                f"Learned scale: **{min(calibration_norms):.2f}-{max(calibration_norms):.2f}**. "
            )
        status = (
            f"Derived **{len(diagnostics['layers'])} live directions** from "
            f"**{diagnostics['exemplar_count']} approved pairs**. "
            f"{calibration_summary}"
            f"Cache hit: **{diagnostics['cache_hit']}**. "
            f"Hooks active after request: **{diagnostics['hooks_active_after_request']}**."
        )
        if warning:
            status = f"**Quality warning:** {escape(str(warning))}\n\n{status}"
        return result["baseline"], result["steered"], diagnostics, status

    def generate_card_core(
        capsule_id: str,
        variation: str,
        principal: Principal | None,
    ) -> tuple[str, str, str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before generating card art.")
        if not capsule_id:
            raise gr.Error("Approve or open a capsule before generating its card.")
        try:
            operations.require(principal.user_id, "card")
            result = card_service.generate(
                principal,
                capsule_id,
                variation=variation,
            )
        except (
            FeatureDisabledError,
            KeyError,
            PermissionError,
            QuotaExceededError,
            ValueError,
        ) as exc:
            raise gr.Error(str(exc)) from exc
        provider_label = (
            "deterministic fallback" if result.used_fallback else "FLUX.2 Klein 4B on Modal"
        )
        status = (
            f"Generated **{escape(result.record.name)}** using **{provider_label}**. "
            f"Composition: **{result.prompt.variation}**. "
            "The visual identity came from this capsule's approved traits and dimensions. "
            "Generate again for a fresh interpretation."
        )
        return (
            str(result.interactive_path),
            str(result.social_path),
            status,
            result.record,
        )

    def create_voice_core(
        capsule_id: str,
        audio_path: str | None,
        signature_text: str,
        consented: bool,
        retention: str,
        principal: Principal | None,
    ) -> tuple[str, str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before creating a voice clone.")
        if not capsule_id:
            raise gr.Error("Approve or open a capsule before creating its voice.")
        try:
            operations.require(principal.user_id, "voice")
            result = voice_service.create_clone(
                principal,
                capsule_id,
                [audio_path] if audio_path else [],
                signature_text,
                consented=consented,
                retention=retention,
            )
        except (
            FeatureDisabledError,
            KeyError,
            PermissionError,
            QuotaExceededError,
            VoiceError,
        ) as exc:
            raise gr.Error(str(exc)) from exc
        lifecycle = (
            f"temporary for {settings.voice_temporary_hours} hours"
            if retention == "temporary"
            else "retained privately until explicit deletion"
        )
        return (
            str(result.audio_path),
            (
                f"Created a private OpenBMB VoxCPM2 voice reference for "
                f"**{escape(result.record.name)}**. The clone is {lifecycle}. "
                "Generated audio is synthetic."
            ),
            result.record,
        )

    def synthesize_voice_core(
        capsule_id: str,
        speech_text: str,
        principal: Principal | None,
    ) -> tuple[str, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before generating speech.")
        if not capsule_id:
            raise gr.Error("Choose a capsule with a voice.")
        try:
            output = voice_service.synthesize(principal, capsule_id, speech_text)
        except (KeyError, PermissionError, VoiceError) as exc:
            raise gr.Error(str(exc)) from exc
        return (
            str(output),
            "Generated with private VoxCPM2 cloning on Modal. This audio is synthetic.",
        )

    def delete_voice_core(
        capsule_id: str,
        principal: Principal | None,
    ) -> tuple[None, str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before deleting a voice.")
        if not capsule_id:
            raise gr.Error("Choose a capsule with a voice.")
        try:
            record = voice_service.delete_voice(principal, capsule_id)
        except (KeyError, PermissionError, VoiceError) as exc:
            raise gr.Error(
                f"{exc} Cleanup is recorded and can be retried without losing the capsule."
            ) from exc
        return None, "VoxCPM2 voice reference deleted. The text capsule remains available.", record

    def publish_selection(
        include_summary: bool,
        include_descriptors: bool,
        include_dimensions: bool,
        include_card: bool,
        include_voice_sample: bool,
    ) -> PublishSelection:
        return PublishSelection(
            include_summary=include_summary,
            include_descriptors=include_descriptors,
            include_dimensions=include_dimensions,
            include_card=include_card,
            include_voice_sample=include_voice_sample,
        )

    def preview_publish_core(
        capsule_id: str,
        include_summary: bool,
        include_descriptors: bool,
        include_dimensions: bool,
        include_card: bool,
        include_voice_sample: bool,
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
                    include_voice_sample,
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
        include_voice_sample: bool,
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
                    include_voice_sample,
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

    def refresh_profile_core(
        capsule_id: str,
        principal: Principal | None,
    ) -> tuple[object, CapsuleRecord, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before refreshing a capsule profile.")
        if not capsule_id:
            raise gr.Error("Choose a saved capsule first.")
        record = capsule_library.get_capsule(principal, capsule_id)
        if not record.exemplar_pairs:
            raise gr.Error("This capsule has no approved examples to analyze.")
        profile = infer_style_profile(
            tuple(MessageRecord(author="You", text=pair.positive) for pair in record.exemplar_pairs)
        )
        updated = capsule_library.save_capsule(
            principal,
            replace(
                record,
                style_profile=profile,
                public_projection=CapsulePublicProjection.from_profile(record.name, profile),
                card_seed=None,
                card_prompt_hash="",
            ),
        )
        return (
            profile.as_dict(include_private_evidence=True),
            updated,
            (
                f"Refreshed **{escape(record.name)}** from "
                f"**{len(record.exemplar_pairs)} approved examples** using the richer "
                "profile model. Generate the card again to apply its new visual identity. "
                "Any already-published page stays unchanged until you publish again."
            ),
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
    ) -> tuple[str, str, object, CapsuleRecord | None]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before deleting a capsule.")
        if not capsule_id:
            raise gr.Error("Choose a capsule to delete.")
        if not confirmed:
            raise gr.Error("Confirm permanent deletion first.")
        try:
            voice_service.delete_voice(principal, capsule_id)
        except VoiceError as exc:
            record = capsule_library.get_capsule(principal, capsule_id)
            return (
                (
                    f"Capsule deletion is paused: {escape(str(exc))} "
                    "The VoxCPM2 cleanup is recorded; retry deletion later."
                ),
                _library_html(principal, capsule_library),
                gr.Dropdown(
                    choices=_capsule_choices(principal, capsule_library),
                    value=capsule_id,
                ),
                record,
            )
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

    def fusion_core(
        first_id: str,
        second_id: str,
        first_percent: float,
        prompt: str,
        name: str,
        voice_strategy: str,
        principal: Principal | None,
    ) -> tuple[str, object, str, str, str, CapsuleRecord, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before fusing capsules.")
        try:
            operations.require(principal.user_id, "fusion")
            result = fusion_service.create(
                principal,
                first_id=first_id,
                second_id=second_id,
                first_weight=float(first_percent) / 100,
                prompt=prompt,
                name=name,
                voice_strategy=voice_strategy,
            )
        except (
            FeatureDisabledError,
            KeyError,
            PermissionError,
            QuotaExceededError,
            SteeringError,
            ValueError,
        ) as exc:
            raise gr.Error(str(exc)) from exc
        return (
            result.response,
            result.diagnostics,
            (
                f"Saved **{escape(result.record.name)}** as a private fusion. "
                "Both source directions were derived and combined inside this request; "
                "no fused tensor was persisted."
            ),
            result.interactive_card_path,
            result.social_card_path,
            result.record,
            _library_html(principal, capsule_library),
        )

    def fusion_compatibility_core(
        first_id: str,
        second_id: str,
        principal: Principal | None,
    ) -> str:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before checking fusion compatibility.")
        try:
            compatibility = fusion_service.compatibility(principal, first_id, second_id)
        except (KeyError, PermissionError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        marker = "Compatible" if compatibility.compatible else "Not compatible"
        return f"**{marker}:** {escape(compatibility.reason)}"

    def battle_core(
        first_id: str,
        second_id: str,
        challenge: str,
        principal: Principal | None,
    ) -> tuple[object, str]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before starting a battle.")
        try:
            operations.require(principal.user_id, "battle")
            result = battle_service.run(
                principal,
                first_id=first_id,
                second_id=second_id,
                challenge=challenge,
            )
        except (
            FeatureDisabledError,
            KeyError,
            PermissionError,
            QuotaExceededError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise gr.Error(str(exc)) from exc
        return (
            result.as_dict(),
            (
                f"Nemotron judged both anonymous orders. Winner: **{escape(result.winner)}**. "
                f"{escape(result.disclaimer)}"
            ),
        )

    def deep_estimate_core(visual_lora: bool) -> tuple[object, str]:
        estimate = estimate_deep_training(visual_lora=visual_lora)
        return (
            estimate.as_dict(),
            (
                f"Estimated **{estimate.estimated_minutes} minutes** and approximately "
                f"**${estimate.estimated_modal_credits:.2f}** of the Modal credit budget."
            ),
        )

    def deep_start_core(
        capsule_id: str,
        idempotency_key: str,
        visual_lora: bool,
        confirmed: bool,
        principal: Principal | None,
    ) -> tuple[object, str, CapsuleRecord]:
        if principal is None:
            raise gr.Error("Sign in with Hugging Face before starting Deep Capsule.")
        try:
            operations.require(principal.user_id, "deep_training")
            record = deep_service.start(
                principal,
                capsule_id,
                idempotency_key=idempotency_key,
                visual_lora=visual_lora,
                confirmed=confirmed,
            )
        except (
            FeatureDisabledError,
            KeyError,
            PermissionError,
            QuotaExceededError,
            ValueError,
        ) as exc:
            raise gr.Error(str(exc)) from exc
        return (
            record.deep_training,
            "Deep Capsule queued on Modal. The Quick Capsule remains fully usable.",
            record,
        )

    def deep_poll_core(
        capsule_id: str,
        principal: Principal | None,
    ) -> tuple[object, str, CapsuleRecord]:
        try:
            record = deep_service.poll(principal, capsule_id)
        except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        job = record.deep_training or {}
        return (
            job,
            f"Deep Capsule status: **{escape(str(job.get('status', 'unknown')))}**.",
            record,
        )

    def deep_cancel_core(
        capsule_id: str,
        principal: Principal | None,
    ) -> tuple[object, str, CapsuleRecord]:
        try:
            record = deep_service.cancel(principal, capsule_id)
        except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        return (
            record.deep_training,
            "Deep Capsule cancelled. The original Quick Capsule is unchanged.",
            record,
        )

    with gr.Blocks(title="Persona Capsule — Your voice, made tangible") as demo:
        gr.HTML(_landing_html(settings))
        draft_state = gr.State(value=None)
        approved_capsule_state = gr.State(value=None)

        with gr.Row(elem_classes=["pc-workspace"]):
            with gr.Column(scale=7, elem_classes=["pc-workspace-intro"]):
                gr.HTML(
                    """
                    <div class="pc-workspace-note">
                      Your active capsule follows you through every step.
                    </div>
                    """
                )
                if settings.oauth_ui_available:
                    gr.LoginButton("Sign in with Hugging Face", elem_classes=["pc-login"])
                else:
                    gr.HTML(
                        """
                        <div class="pc-login-unavailable">
                          Using the configured local development identity.
                          Hugging Face OAuth activates in the deployed Space.
                        </div>
                        """
                    )
                account = gr.HTML()
            with gr.Column(scale=13):
                with gr.Group(elem_classes=["pc-active-capsule"]):
                    gr.Markdown("### Active capsule")
                    with gr.Row():
                        capsule_selector = gr.Dropdown(
                            label="Choose from your private library",
                            choices=[],
                        )
                        open_capsule = gr.Button("Use this capsule")
                    lifecycle_status = gr.Markdown(
                        "Choose a saved capsule, or create one in step 1."
                    )
                library = gr.HTML()

        with gr.Tabs(
            selected="create",
            elem_classes=["pc-journey-tabs"],
        ) as journey_tabs:
            with gr.Tab("1 · Create", id="create"):
                with gr.Column(elem_classes=["pc-tab-shell"]):
                    gr.HTML(
                        """
                        <section class="pc-section-intro">
                          <div>
                            <span class="pc-step-chip">Step 1 · about 2 minutes</span>
                            <h3>Create your capsule.</h3>
                          </div>
                          <p>
                            Paste at least eight messages that sound like you. We remove
                            obvious private details, infer communication patterns, and ask
                            you to approve the small set used for live steering.
                          </p>
                        </section>
                        """
                    )
                    with gr.Group(elem_classes=["pc-primary-panel"]):
                        with gr.Row():
                            capsule_name = gr.Textbox(
                                label="1. Name your capsule",
                                placeholder="e.g. Clear Signal",
                            )
                            speaker = gr.Textbox(
                                label="Your name in the messages",
                                value="You",
                                info='For "Name: message" exports, enter that name.',
                            )
                        raw_messages = gr.Textbox(
                            label="2. Paste messages that represent how you communicate",
                            lines=10,
                            placeholder=(
                                "You: Thanks — I see the tradeoff. Let’s test the smallest "
                                "version first.\n"
                                "You: I’m not convinced yet; what would change your mind?\n"
                                "…paste at least 8 varied messages, ideally around 20."
                            ),
                        )
                        consent = gr.Checkbox(
                            label="I own these messages or have permission to process them.",
                            value=False,
                        )
                        gr.HTML(
                            """
                            <div class="pc-help">
                              Your original paste exists only while you review this draft.
                              Approval keeps the profile and selected contrast pairs, then
                              discards the original and every unselected message.
                            </div>
                            """
                        )
                        analyze = gr.Button(
                            "Analyze my communication style",
                            elem_classes=["pc-button"],
                        )
                        creation_status = gr.Markdown(elem_classes=["pc-result-panel"])
                        cleaned_preview = gr.Markdown()
                        pair_table = gr.Dataframe(
                            headers=["Keep", "Sounds like me", "Neutral comparison"],
                            datatype=["bool", "str", "str"],
                            interactive=True,
                            label="3. Choose the examples that best represent you",
                        )
                        with gr.Accordion(
                            "Fine-tune the inferred profile (optional)",
                            open=False,
                            elem_classes=["pc-advanced"],
                        ):
                            profile_json = gr.JSON(label="Profile evidence")
                            gr.Markdown(
                                "Adjust only what feels inaccurate. These are style controls, "
                                "not a clinical or psychological assessment."
                            )
                            with gr.Row():
                                openness = gr.Slider(0, 100, value=50, label="Openness")
                                conscientiousness = gr.Slider(
                                    0,
                                    100,
                                    value=50,
                                    label="Conscientiousness",
                                )
                                expressiveness = gr.Slider(
                                    0,
                                    100,
                                    value=50,
                                    label="Expressiveness",
                                )
                            with gr.Row():
                                agreeableness = gr.Slider(
                                    0,
                                    100,
                                    value=50,
                                    label="Agreeableness",
                                )
                                emotional_range = gr.Slider(
                                    0,
                                    100,
                                    value=50,
                                    label="Emotional range",
                                )
                                directness = gr.Slider(0, 100, value=50, label="Directness")
                                formality = gr.Slider(0, 100, value=50, label="Formality")
                        approve = gr.Button(
                            "Approve capsule and continue to testing",
                            elem_classes=["pc-button"],
                        )

                    with gr.Accordion(
                        "Manage, export, or delete saved capsules",
                        open=False,
                        elem_classes=["pc-advanced", "pc-danger-zone"],
                    ):
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
                        refresh_profile = gr.Button(
                            "Refresh personality labels from approved examples"
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

            with gr.Tab("2 · Test", id="test"):
                with gr.Column(elem_classes=["pc-tab-shell"]):
                    gr.HTML(
                        """
                        <section class="pc-section-intro">
                          <div>
                            <span class="pc-step-chip">Step 2 · the signature demo</span>
                            <h3>See your steering vector work.</h3>
                          </div>
                          <p>
                            Ask one question and compare MiniCPM's normal answer with
                            the same model steered by your approved capsule. Start at
                            0.85; move the slider only after reading the first result.
                          </p>
                        </section>
                        """
                    )
                    with gr.Group(elem_classes=["pc-primary-panel"]):
                        live_prompt = gr.Textbox(
                            label="What should both versions answer?",
                            value=(
                                "Explain why a small team should test the riskiest "
                                "assumption before building the full product."
                            ),
                            lines=3,
                        )
                        live_strength = gr.Slider(
                            -1.5,
                            1.5,
                            value=0.85,
                            step=0.05,
                            label="How strongly should the capsule steer the response?",
                            info="0 is unchanged. Values above ±1.1 may reduce coherence.",
                        )
                        live_run = gr.Button(
                            "Run the before-and-after comparison",
                            elem_classes=["pc-button"],
                        )
                        live_status = gr.Markdown(elem_classes=["pc-result-panel"])
                        with gr.Row():
                            baseline_output = gr.Textbox(
                                label="Before · MiniCPM baseline",
                                lines=9,
                            )
                            steered_output = gr.Textbox(
                                label="After · your live-steered response",
                                lines=9,
                            )
                    with gr.Accordion(
                        "How the live vector was applied",
                        open=False,
                        elem_classes=["pc-advanced"],
                    ):
                        gr.Markdown(
                            "These diagnostics prove the directions were derived for this "
                            "request, applied across five layers, and removed afterward."
                        )
                        vector_diagnostics = gr.JSON(
                            label="Request-scoped vector diagnostics",
                        )

                    with gr.Accordion(
                        "Try the offline sample capsule",
                        open=False,
                        elem_classes=["pc-advanced"],
                    ):
                        gr.Markdown(
                            "No provider call. Use this only to understand the interaction "
                            "while a model is cold or unavailable."
                        )
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
                        run = gr.Button(
                            "Respond as the sample capsule",
                            elem_classes=["pc-button"],
                        )
                        output = gr.Markdown(label="Capsule response")
                        run.click(demo_reply, inputs=[prompt, intensity], outputs=output)

                    with gr.Accordion(
                        "Fuse two capsules",
                        open=False,
                        elem_classes=["pc-advanced"],
                    ):
                        gr.Markdown(
                            "Combine two compatible personality directions into a new, "
                            "editable private capsule."
                        )
                        with gr.Row():
                            fusion_source_a = gr.Dropdown(label="Fusion source A", choices=[])
                            fusion_source_b = gr.Dropdown(label="Fusion source B", choices=[])
                        fusion_weight = gr.Slider(
                            0,
                            100,
                            value=50,
                            step=5,
                            label="Source A percentage",
                        )
                        fusion_name = gr.Textbox(
                            label="Fused capsule name",
                            placeholder="e.g. Clear Spark",
                        )
                        fusion_prompt = gr.Textbox(
                            label="Shared generation prompt",
                            placeholder="Explain the next useful experiment.",
                            lines=3,
                        )
                        fusion_voice = gr.Radio(
                            choices=["none", "source_a", "source_b", "alternate"],
                            value="none",
                            label="Speaking voice strategy",
                            info=(
                                "Voices are selected from permitted sources, "
                                "never mathematically blended."
                            ),
                        )
                        with gr.Row():
                            check_fusion = gr.Button("Check compatibility")
                            create_fusion = gr.Button(
                                "Generate and save fusion",
                                elem_classes=["pc-button"],
                            )
                        fusion_status = gr.Markdown("Choose two available source capsules.")
                        fusion_response = gr.Textbox(label="Fused MiniCPM response", lines=8)
                        fusion_diagnostics = gr.JSON(label="Fusion diagnostics")
                        with gr.Row():
                            fusion_card = gr.Image(
                                label="Fused collectible card",
                                type="filepath",
                            )
                            fusion_social = gr.Image(
                                label="Fused social preview",
                                type="filepath",
                            )

                    with gr.Accordion(
                        "Battle two capsules with Nemotron",
                        open=False,
                        elem_classes=["pc-advanced"],
                    ):
                        gr.Markdown(
                            "MiniCPM answers once per capsule. Nemotron judges both "
                            "anonymous orders for style, quality, adherence, and safety."
                        )
                        with gr.Row():
                            battle_source_a = gr.Dropdown(label="Battle capsule A", choices=[])
                            battle_source_b = gr.Dropdown(label="Battle capsule B", choices=[])
                        battle_challenge = gr.Textbox(
                            label="Battle challenge",
                            value=(
                                "Explain why a small team should test the riskiest assumption "
                                "before committing to a large launch."
                            ),
                            lines=3,
                        )
                        run_battle = gr.Button(
                            "Run blinded Nemotron battle",
                            elem_classes=["pc-button"],
                        )
                        battle_status = gr.Markdown()
                        battle_result = gr.JSON(label="Order-swapped game result")

            with gr.Tab("3 · Bring it to life", id="share"):
                with gr.Column(elem_classes=["pc-tab-shell"]):
                    gr.HTML(
                        """
                        <section class="pc-section-intro">
                          <div>
                            <span class="pc-step-chip">Step 3 · make it tangible</span>
                            <h3>See it. Hear it. Share it.</h3>
                          </div>
                          <p>
                            Start with the collectible card. Voice cloning and public
                            sharing remain separate, explicit choices so nothing leaves
                            your private workspace by accident.
                          </p>
                        </section>
                        """
                    )
                    with gr.Group(elem_classes=["pc-primary-panel"]):
                        gr.Markdown("### Generate the collectible card")
                        gr.Markdown(
                            "FLUX receives approved descriptors and style dimensions, "
                            "never your private messages."
                        )
                        card_variation = gr.Dropdown(
                            choices=[
                                ("Focused portrait", "signal"),
                                ("Layered archive", "archive"),
                                ("Dynamic motion", "kinetic"),
                            ],
                            value="signal",
                            label="Composition",
                            info=(
                                "Choose the framing only. Persona traits automatically control "
                                "the character, wardrobe, palette, pose, and symbolic setting."
                            ),
                        )
                        generate_card = gr.Button(
                            "Generate my collectible card",
                            elem_classes=["pc-button"],
                        )
                        card_status = gr.Markdown(elem_classes=["pc-result-panel"])
                        with gr.Row():
                            interactive_card = gr.Image(
                                label="Capsule card · 768×1024",
                                type="filepath",
                            )
                            social_card = gr.Image(
                                label="X preview · 1200×628",
                                type="filepath",
                            )

                    with gr.Accordion(
                        "Add an authorized VoxCPM2 voice",
                        open=False,
                        visible=settings.enable_voice and settings.voxcpm_available,
                        elem_classes=["pc-advanced"],
                    ):
                        gr.Markdown(
                            "Use only your own voice or a recording you have explicit "
                            "permission to clone. Use 1–2 minutes of clean, consistent "
                            "speech with no music, reverb, or other speakers. Instant Voice "
                            "The private reference is processed by OpenBMB VoxCPM2 on Modal "
                            "and can be deleted at any time."
                        )
                        voice_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Authorized voice recording",
                        )
                        voice_consent = gr.Checkbox(
                            label="I own this voice or have explicit permission to clone it.",
                            value=False,
                        )
                        voice_retention = gr.Radio(
                            choices=["temporary", "retained"],
                            value="temporary",
                            label="Clone lifecycle",
                            info=(
                                f"Temporary clones expire after "
                                f"{settings.voice_temporary_hours} hours."
                            ),
                        )
                        signature_line = gr.Textbox(
                            label="First line to speak",
                            value="Small steps, clear signal, real momentum.",
                            lines=2,
                        )
                        create_voice = gr.Button(
                            "Create VoxCPM2 voice",
                            elem_classes=["pc-button"],
                        )
                        voice_status = gr.Markdown()
                        voice_output = gr.Audio(
                            label="Synthetic capsule speech",
                            type="filepath",
                        )
                        speech_text = gr.Textbox(
                            label="Speak another response",
                            placeholder="Enter up to 1000 characters.",
                            lines=3,
                        )
                        with gr.Row():
                            synthesize_voice = gr.Button("Generate synthetic speech")
                            delete_voice = gr.Button("Delete VoxCPM2 voice")

                    with gr.Accordion(
                        "Preview and publish a shareable capsule",
                        open=False,
                        elem_classes=["pc-advanced"],
                    ):
                        gr.Markdown(
                            "Choose exactly what becomes public. Private messages, "
                            "evidence, owner IDs, and provider references never travel."
                        )
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
                            publish_voice = gr.Checkbox(
                                label="Public synthetic voice sample",
                                value=False,
                            )
                        preview_publish = gr.Button("Preview exactly what will be public")
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

            # Keep the resumable training controls wired for internal development,
            # but do not expose Deep Capsule until its adapter is served in live chat.
            with gr.Tab("Advanced lab", id="lab", visible=False):
                with gr.Column(elem_classes=["pc-tab-shell"]):
                    gr.HTML(
                        """
                        <section class="pc-section-intro">
                          <div>
                            <span class="pc-step-chip">Optional · compute intensive</span>
                            <h3>Train a Deep Capsule.</h3>
                          </div>
                          <p>
                            The Quick Capsule already works without training. Use this
                            asynchronous Modal job only when you have reviewed the estimate
                            and want a private adapter trained and evaluated against held-out
                            data. A passing adapter is saved privately; the current live chat
                            continues to use request-scoped Quick Capsule steering.
                          </p>
                        </section>
                        """
                    )
                    with gr.Group(elem_classes=["pc-primary-panel"]):
                        deep_visual_lora = gr.Checkbox(
                            label="Also request optional personal visual LoRA",
                            value=False,
                            info=(
                                "Requires a separately reviewed image dataset. "
                                "Without one, only the writing adapter is requested."
                            ),
                        )
                        deep_idempotency_key = gr.Textbox(
                            label="Job idempotency key",
                            value=f"deep-{uuid4().hex}",
                        )
                        estimate_deep = gr.Button("Estimate time and Modal credits")
                        deep_status = gr.Markdown(elem_classes=["pc-result-panel"])
                        deep_job = gr.JSON(label="Resumable job state")
                        deep_confirmation = gr.Checkbox(
                            label=("I reviewed the estimate and want to start this Modal job."),
                            value=False,
                        )
                        with gr.Row():
                            start_deep = gr.Button(
                                "Start Deep Capsule",
                                elem_classes=["pc-button"],
                            )
                            poll_deep = gr.Button("Refresh job status")
                            cancel_deep = gr.Button("Cancel job")

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
            raw_messages,
            speaker,
            consent,
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
            lifecycle_status,
            raw_messages,
            cleaned_preview,
            draft_state,
            approved_capsule_state,
            capsule_selector,
            journey_tabs,
        ]
        estimate_deep.click(
            deep_estimate_core,
            inputs=[deep_visual_lora],
            outputs=[deep_job, deep_status],
        )

        if settings.oauth_ui_available:

            def load_private_library(
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, object, object, object, object, object, CapsuleRecord | None]:
                principal = identity_gateway.resolve(profile)
                voice_service.cleanup_expired(principal)
                choices = _capsule_choices(principal, capsule_library)
                steerable_choices = _steerable_capsule_choices(principal, capsule_library)
                selected_value = choices[0][1] if choices else None
                selected_capsule = (
                    capsule_library.get_capsule(principal, selected_value)
                    if principal is not None and selected_value
                    else None
                )
                first_value = steerable_choices[0][1] if steerable_choices else None
                second_value = steerable_choices[1][1] if len(steerable_choices) > 1 else None
                return (
                    _account_html(principal),
                    _library_html(principal, capsule_library),
                    gr.Dropdown(choices=choices, value=selected_value),
                    gr.Dropdown(choices=steerable_choices, value=first_value),
                    gr.Dropdown(choices=steerable_choices, value=second_value),
                    gr.Dropdown(choices=steerable_choices, value=first_value),
                    gr.Dropdown(choices=steerable_choices, value=second_value),
                    selected_capsule,
                )

            demo.load(
                load_private_library,
                inputs=None,
                outputs=[
                    account,
                    library,
                    capsule_selector,
                    fusion_source_a,
                    fusion_source_b,
                    battle_source_a,
                    battle_source_b,
                    approved_capsule_state,
                ],
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
                raw_input: str,
                speaker_label: str,
                has_consent: bool,
                rows: object,
                open_value: float,
                conscientious_value: float,
                expressive_value: float,
                agreeable_value: float,
                emotional_value: float,
                direct_value: float,
                formal_value: float,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, str, str, str, None, CapsuleRecord, object, object]:
                return approve_core(
                    name,
                    draft,
                    raw_input,
                    speaker_label,
                    has_consent,
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

            def fusion_with_oauth(
                first_id: str,
                second_id: str,
                first_percent: float,
                prompt_value: str,
                name: str,
                voice_strategy: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, object, str, str, str, CapsuleRecord, str]:
                return fusion_core(
                    first_id,
                    second_id,
                    first_percent,
                    prompt_value,
                    name,
                    voice_strategy,
                    identity_gateway.resolve(profile),
                )

            def fusion_compatibility_with_oauth(
                first_id: str,
                second_id: str,
                profile: gr.OAuthProfile | None,
            ) -> str:
                return fusion_compatibility_core(
                    first_id,
                    second_id,
                    identity_gateway.resolve(profile),
                )

            def battle_with_oauth(
                first_id: str,
                second_id: str,
                challenge: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, str]:
                return battle_core(
                    first_id,
                    second_id,
                    challenge,
                    identity_gateway.resolve(profile),
                )

            def deep_start_with_oauth(
                capsule_id: str,
                idempotency_key: str,
                visual_lora: bool,
                confirmed: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, str, CapsuleRecord]:
                return deep_start_core(
                    capsule_id,
                    idempotency_key,
                    visual_lora,
                    confirmed,
                    identity_gateway.resolve(profile),
                )

            def deep_poll_with_oauth(
                capsule_id: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, str, CapsuleRecord]:
                return deep_poll_core(capsule_id, identity_gateway.resolve(profile))

            def deep_cancel_with_oauth(
                capsule_id: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, str, CapsuleRecord]:
                return deep_cancel_core(capsule_id, identity_gateway.resolve(profile))

            check_fusion.click(
                fusion_compatibility_with_oauth,
                inputs=[fusion_source_a, fusion_source_b],
                outputs=[fusion_status],
            )
            create_fusion.click(
                fusion_with_oauth,
                inputs=[
                    fusion_source_a,
                    fusion_source_b,
                    fusion_weight,
                    fusion_prompt,
                    fusion_name,
                    fusion_voice,
                ],
                outputs=[
                    fusion_response,
                    fusion_diagnostics,
                    fusion_status,
                    fusion_card,
                    fusion_social,
                    approved_capsule_state,
                    library,
                ],
            )
            run_battle.click(
                battle_with_oauth,
                inputs=[battle_source_a, battle_source_b, battle_challenge],
                outputs=[battle_result, battle_status],
            )
            start_deep.click(
                deep_start_with_oauth,
                inputs=[
                    capsule_selector,
                    deep_idempotency_key,
                    deep_visual_lora,
                    deep_confirmation,
                ],
                outputs=[deep_job, deep_status, approved_capsule_state],
            )
            poll_deep.click(
                deep_poll_with_oauth,
                inputs=[capsule_selector],
                outputs=[deep_job, deep_status, approved_capsule_state],
            )
            cancel_deep.click(
                deep_cancel_with_oauth,
                inputs=[capsule_selector],
                outputs=[deep_job, deep_status, approved_capsule_state],
            )

            def generate_card_with_oauth(
                capsule_id: str,
                variation: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, str, CapsuleRecord]:
                return generate_card_core(
                    capsule_id,
                    variation,
                    identity_gateway.resolve(profile),
                )

            generate_card.click(
                generate_card_with_oauth,
                inputs=[capsule_selector, card_variation],
                outputs=[
                    interactive_card,
                    social_card,
                    card_status,
                    approved_capsule_state,
                ],
            )

            def create_voice_with_oauth(
                capsule_id: str,
                audio_path: str | None,
                text: str,
                consented: bool,
                retention: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str, CapsuleRecord]:
                return create_voice_core(
                    capsule_id,
                    audio_path,
                    text,
                    consented,
                    retention,
                    identity_gateway.resolve(profile),
                )

            def synthesize_voice_with_oauth(
                capsule_id: str,
                text: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str]:
                return synthesize_voice_core(
                    capsule_id,
                    text,
                    identity_gateway.resolve(profile),
                )

            def delete_voice_with_oauth(
                capsule_id: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[None, str, CapsuleRecord]:
                return delete_voice_core(
                    capsule_id,
                    identity_gateway.resolve(profile),
                )

            create_voice.click(
                create_voice_with_oauth,
                inputs=[
                    capsule_selector,
                    voice_audio,
                    signature_line,
                    voice_consent,
                    voice_retention,
                ],
                outputs=[voice_output, voice_status, approved_capsule_state],
            )
            synthesize_voice.click(
                synthesize_voice_with_oauth,
                inputs=[capsule_selector, speech_text],
                outputs=[voice_output, voice_status],
            )
            delete_voice.click(
                delete_voice_with_oauth,
                inputs=[capsule_selector],
                outputs=[voice_output, voice_status, approved_capsule_state],
            )

            def preview_publish_with_oauth(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                voice: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, str]:
                return preview_publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    voice,
                    identity_gateway.resolve(profile),
                )

            def publish_with_oauth(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                voice: bool,
                confirmed: bool,
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, CapsuleRecord]:
                return publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    voice,
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
                publish_voice,
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

            def refresh_with_oauth(
                capsule_id: str,
                profile: gr.OAuthProfile | None,
            ) -> tuple[object, CapsuleRecord, str]:
                return refresh_profile_core(
                    capsule_id,
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
            refresh_profile.click(
                refresh_with_oauth,
                inputs=[capsule_selector],
                outputs=[
                    reopened_profile,
                    approved_capsule_state,
                    lifecycle_status,
                ],
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

            def load_local_library() -> tuple[
                str,
                str,
                object,
                object,
                object,
                object,
                object,
                CapsuleRecord | None,
            ]:
                principal = identity_gateway.resolve_local()
                voice_service.cleanup_expired(principal)
                choices = _capsule_choices(principal, capsule_library)
                steerable_choices = _steerable_capsule_choices(principal, capsule_library)
                selected_value = choices[0][1] if choices else None
                selected_capsule = (
                    capsule_library.get_capsule(principal, selected_value)
                    if principal is not None and selected_value
                    else None
                )
                first_value = steerable_choices[0][1] if steerable_choices else None
                second_value = steerable_choices[1][1] if len(steerable_choices) > 1 else None
                return (
                    _account_html(principal),
                    _library_html(principal, capsule_library),
                    gr.Dropdown(choices=choices, value=selected_value),
                    gr.Dropdown(choices=steerable_choices, value=first_value),
                    gr.Dropdown(choices=steerable_choices, value=second_value),
                    gr.Dropdown(choices=steerable_choices, value=first_value),
                    gr.Dropdown(choices=steerable_choices, value=second_value),
                    selected_capsule,
                )

            demo.load(
                load_local_library,
                inputs=None,
                outputs=[
                    account,
                    library,
                    capsule_selector,
                    fusion_source_a,
                    fusion_source_b,
                    battle_source_a,
                    battle_source_b,
                    approved_capsule_state,
                ],
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
                raw_input: str,
                speaker_label: str,
                has_consent: bool,
                rows: object,
                open_value: float,
                conscientious_value: float,
                expressive_value: float,
                agreeable_value: float,
                emotional_value: float,
                direct_value: float,
                formal_value: float,
            ) -> tuple[str, str, str, str, str, None, CapsuleRecord, object, object]:
                return approve_core(
                    name,
                    draft,
                    raw_input,
                    speaker_label,
                    has_consent,
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

            def fusion_locally(
                first_id: str,
                second_id: str,
                first_percent: float,
                prompt_value: str,
                name: str,
                voice_strategy: str,
            ) -> tuple[str, object, str, str, str, CapsuleRecord, str]:
                return fusion_core(
                    first_id,
                    second_id,
                    first_percent,
                    prompt_value,
                    name,
                    voice_strategy,
                    identity_gateway.resolve_local(),
                )

            def fusion_compatibility_locally(
                first_id: str,
                second_id: str,
            ) -> str:
                return fusion_compatibility_core(
                    first_id,
                    second_id,
                    identity_gateway.resolve_local(),
                )

            def battle_locally(
                first_id: str,
                second_id: str,
                challenge: str,
            ) -> tuple[object, str]:
                return battle_core(
                    first_id,
                    second_id,
                    challenge,
                    identity_gateway.resolve_local(),
                )

            def deep_start_locally(
                capsule_id: str,
                idempotency_key: str,
                visual_lora: bool,
                confirmed: bool,
            ) -> tuple[object, str, CapsuleRecord]:
                return deep_start_core(
                    capsule_id,
                    idempotency_key,
                    visual_lora,
                    confirmed,
                    identity_gateway.resolve_local(),
                )

            def deep_poll_locally(
                capsule_id: str,
            ) -> tuple[object, str, CapsuleRecord]:
                return deep_poll_core(capsule_id, identity_gateway.resolve_local())

            def deep_cancel_locally(
                capsule_id: str,
            ) -> tuple[object, str, CapsuleRecord]:
                return deep_cancel_core(capsule_id, identity_gateway.resolve_local())

            check_fusion.click(
                fusion_compatibility_locally,
                inputs=[fusion_source_a, fusion_source_b],
                outputs=[fusion_status],
            )
            create_fusion.click(
                fusion_locally,
                inputs=[
                    fusion_source_a,
                    fusion_source_b,
                    fusion_weight,
                    fusion_prompt,
                    fusion_name,
                    fusion_voice,
                ],
                outputs=[
                    fusion_response,
                    fusion_diagnostics,
                    fusion_status,
                    fusion_card,
                    fusion_social,
                    approved_capsule_state,
                    library,
                ],
            )
            run_battle.click(
                battle_locally,
                inputs=[battle_source_a, battle_source_b, battle_challenge],
                outputs=[battle_result, battle_status],
            )
            start_deep.click(
                deep_start_locally,
                inputs=[
                    capsule_selector,
                    deep_idempotency_key,
                    deep_visual_lora,
                    deep_confirmation,
                ],
                outputs=[deep_job, deep_status, approved_capsule_state],
            )
            poll_deep.click(
                deep_poll_locally,
                inputs=[capsule_selector],
                outputs=[deep_job, deep_status, approved_capsule_state],
            )
            cancel_deep.click(
                deep_cancel_locally,
                inputs=[capsule_selector],
                outputs=[deep_job, deep_status, approved_capsule_state],
            )

            def generate_card_locally(
                capsule_id: str,
                variation: str,
            ) -> tuple[str, str, str, CapsuleRecord]:
                return generate_card_core(
                    capsule_id,
                    variation,
                    identity_gateway.resolve_local(),
                )

            generate_card.click(
                generate_card_locally,
                inputs=[capsule_selector, card_variation],
                outputs=[
                    interactive_card,
                    social_card,
                    card_status,
                    approved_capsule_state,
                ],
            )

            def create_voice_locally(
                capsule_id: str,
                audio_path: str | None,
                text: str,
                consented: bool,
                retention: str,
            ) -> tuple[str, str, CapsuleRecord]:
                return create_voice_core(
                    capsule_id,
                    audio_path,
                    text,
                    consented,
                    retention,
                    identity_gateway.resolve_local(),
                )

            def synthesize_voice_locally(
                capsule_id: str,
                text: str,
            ) -> tuple[str, str]:
                return synthesize_voice_core(
                    capsule_id,
                    text,
                    identity_gateway.resolve_local(),
                )

            def delete_voice_locally(
                capsule_id: str,
            ) -> tuple[None, str, CapsuleRecord]:
                return delete_voice_core(
                    capsule_id,
                    identity_gateway.resolve_local(),
                )

            create_voice.click(
                create_voice_locally,
                inputs=[
                    capsule_selector,
                    voice_audio,
                    signature_line,
                    voice_consent,
                    voice_retention,
                ],
                outputs=[voice_output, voice_status, approved_capsule_state],
            )
            synthesize_voice.click(
                synthesize_voice_locally,
                inputs=[capsule_selector, speech_text],
                outputs=[voice_output, voice_status],
            )
            delete_voice.click(
                delete_voice_locally,
                inputs=[capsule_selector],
                outputs=[voice_output, voice_status, approved_capsule_state],
            )

            def preview_publish_locally(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                voice: bool,
            ) -> tuple[object, str]:
                return preview_publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    voice,
                    identity_gateway.resolve_local(),
                )

            def publish_locally(
                capsule_id: str,
                summary: bool,
                descriptors: bool,
                dimensions: bool,
                card: bool,
                voice: bool,
                confirmed: bool,
            ) -> tuple[str, CapsuleRecord]:
                return publish_core(
                    capsule_id,
                    summary,
                    descriptors,
                    dimensions,
                    card,
                    voice,
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
                publish_voice,
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

            def refresh_locally(
                capsule_id: str,
            ) -> tuple[object, CapsuleRecord, str]:
                return refresh_profile_core(
                    capsule_id,
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
            refresh_profile.click(
                refresh_locally,
                inputs=[capsule_selector],
                outputs=[
                    reopened_profile,
                    approved_capsule_state,
                    lifecycle_status,
                ],
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
