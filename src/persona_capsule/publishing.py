"""Explicit public projections, social metadata, and reversible publishing."""

import json
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
            projection["voice_label"] = "Synthetic voice generated with OpenBMB VoxCPM2"
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
        if projection.get("social_image"):
            self._persist_artifact(record, record.social_image_ref)
            if record.card_image_ref:
                try:
                    self._persist_artifact(record, record.card_image_ref)
                except (CapsuleNotFoundError, FileNotFoundError):
                    # Older capsules may not have persisted the portrait card.
                    # Their social image remains a valid public fallback.
                    pass
        if projection.get("voice_sample"):
            self._persist_artifact(record, record.voice_sample_ref)
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
        text = (
            f"I turned my communication style into {record.name}, a live-steered "
            'Persona Capsule. Chat with it, then try "Do you really know me?"'
        )
        return f"https://x.com/intent/post?text={quote(text)}&url={quote(url)}"

    def get_public(self, slug: str) -> CapsuleRecord:
        if not slug or len(slug) > 64:
            raise CapsuleNotFoundError(slug)
        record = self._repository.get_public_by_slug(slug)
        if record.published_projection is None:
            raise CapsuleNotFoundError(slug)
        return record

    def _local_artifact_path(self, record: CapsuleRecord, reference: str) -> Path:
        owner_namespace = sha256(record.owner_id.encode()).hexdigest()[:24]
        return (
            self._artifact_root
            / "artifacts"
            / owner_namespace
            / record.capsule_id
            / Path(reference).name
        )

    def _persist_artifact(self, record: CapsuleRecord, reference: str) -> None:
        source_path = self._local_artifact_path(record, reference)
        if not source_path.is_file():
            raise CapsuleNotFoundError(reference)
        self._repository.persist_artifact(
            record.owner_id,
            record.capsule_id,
            reference,
            source_path,
        )

    def _resolve_artifact(self, record: CapsuleRecord, reference: str) -> Path:
        local_path = self._local_artifact_path(record, reference)
        if local_path.is_file():
            return local_path
        return self._repository.resolve_artifact(
            record.owner_id,
            record.capsule_id,
            reference,
        )

    def public_image_path(self, record: CapsuleRecord) -> Path:
        projection = record.published_projection or {}
        if not projection.get("social_image") or not record.social_image_ref:
            raise CapsuleNotFoundError(record.public_slug)
        return self._resolve_artifact(record, record.social_image_ref)

    def public_card_path(self, record: CapsuleRecord) -> Path:
        projection = record.published_projection or {}
        if not projection.get("social_image"):
            raise CapsuleNotFoundError(record.public_slug)
        if record.card_image_ref:
            try:
                return self._resolve_artifact(record, record.card_image_ref)
            except (CapsuleNotFoundError, FileNotFoundError):
                pass
        return self.public_image_path(record)

    def public_audio_path(self, record: CapsuleRecord) -> Path:
        projection = record.published_projection or {}
        if not projection.get("voice_sample") or not record.voice_sample_ref:
            raise CapsuleNotFoundError(record.public_slug)
        return self._resolve_artifact(record, record.voice_sample_ref)

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
        card_url = f"{canonical}/card"
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
            f'<div class="card-stage"><img src="{escape(card_url)}" '
            f'alt="{escape(title)} generated card"></div>'
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
        chat_endpoint = json.dumps(f"{canonical}/chat")
        challenge_endpoint = json.dumps(f"{canonical}/challenge")
        challenge_guess_endpoint = json.dumps(f"{canonical}/challenge/guess")
        canonical_json = json.dumps(canonical)
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
    body {{ margin:0; background:var(--ink); color:var(--paper);
      font-family:"Avenir Next","Gill Sans",sans-serif; }}
    body::before {{ background:linear-gradient(90deg,rgba(243,239,226,.025) 1px,
      transparent 1px),linear-gradient(rgba(243,239,226,.025) 1px,transparent 1px);
      background-size:32px 32px; content:""; inset:0; pointer-events:none;
      position:fixed; }}
    main {{ margin:auto; max-width:1240px; padding:28px 34px 80px; position:relative; }}
    nav {{ align-items:center; border-bottom:1px solid #35352e; display:flex;
      justify-content:space-between; padding:12px 0 18px; }}
    nav b {{ font:700 22px Georgia,serif; }}
    nav span {{ font-size:10px; font-weight:900; letter-spacing:.16em;
      text-transform:uppercase; }}
    article {{ border:1px solid #4a4a41; display:grid;
      grid-template-columns:minmax(0,.86fr) minmax(420px,1.14fr); margin-top:42px; }}
    .card-stage {{ background:#0f0f0c; border-right:1px solid #4a4a41;
      display:grid; min-width:0; overflow:hidden; place-items:center; width:100%; }}
    .card-stage img {{ display:block; height:auto; max-height:none; object-fit:contain;
      object-position:center top; width:100%; }}
    .identity {{ align-content:center; padding:52px; }}
    .eyebrow {{ color:var(--acid); font-size:11px; font-weight:900;
      letter-spacing:.16em; text-transform:uppercase; }}
    h1 {{ font:500 clamp(52px,6vw,86px)/.88 Georgia,serif; letter-spacing:-.055em;
      margin:18px 0 24px; }}
    h2 {{ font:500 clamp(36px,4vw,58px)/.94 Georgia,serif; letter-spacing:-.04em;
      margin:12px 0 18px; }}
    p {{ font-size:18px; line-height:1.55; }}
    .traits {{ display:flex; flex-wrap:wrap; gap:8px; margin:28px 0; }}
    .traits span {{ border:1px solid #67675d; border-radius:999px; font-size:12px;
      padding:8px 12px; }}
    ul {{ list-style:none; padding:0; }}
    li {{ display:flex; justify-content:space-between; border-top:1px solid #555;
      padding:12px 0; }}
    a.share {{ display:inline-block; background:var(--acid); color:var(--ink);
      padding:14px 18px; font-weight:800; text-decoration:none; margin-top:24px; }}
    small {{ display:block; margin-top:24px; opacity:.7; }}
    .voice {{ border-top:1px solid #555; margin-top:28px; padding-top:20px; }}
    .voice audio {{ display:block; margin-top:12px; max-width:100%; }}
    .voice small {{ margin-top:8px; }}
    .challenge-section {{ background:var(--rust); color:var(--paper); display:grid;
      gap:42px; grid-template-columns:minmax(260px,.72fr) minmax(0,1.28fr);
      margin-top:18px; padding:34px; }}
    .challenge-section .eyebrow {{ color:var(--ink); }}
    .challenge-section h2 {{ margin-bottom:12px; }}
    .challenge-section p {{ margin:0; }}
    .challenge-panel {{ background:var(--paper); color:var(--ink); padding:22px; }}
    .challenge-start {{ min-height:50px; padding:12px 18px; width:100%; }}
    .challenge-content[hidden], .challenge-share[hidden] {{ display:none; }}
    .challenge-prompt {{ border-left:4px solid var(--rust); font:700 18px/1.4 Georgia,serif;
      margin:0 0 16px; padding-left:14px; }}
    .challenge-answers {{ display:grid; gap:10px; grid-template-columns:1fr 1fr; }}
    .challenge-answer {{ background:#fffdf4; color:var(--ink); font:inherit;
      letter-spacing:0; min-height:150px; padding:16px; text-align:left;
      text-transform:none; }}
    .challenge-answer::before {{ display:block; font-size:11px; font-weight:900;
      letter-spacing:.12em; margin-bottom:10px; text-transform:uppercase; }}
    .challenge-answer[data-index="0"]::before {{ content:"Answer A"; }}
    .challenge-answer[data-index="1"]::before {{ content:"Answer B"; }}
    .challenge-answer.correct {{ background:var(--acid); }}
    .challenge-answer.wrong {{ background:var(--rust); color:var(--paper); }}
    .challenge-result {{ font-weight:800; min-height:24px; }}
    .challenge-share {{ background:var(--ink) !important; color:var(--paper) !important;
      margin-top:10px !important; }}
    .chat-section {{ border-top:1px solid #4a4a41; display:grid; gap:50px;
      grid-template-columns:minmax(260px,.72fr) minmax(0,1.28fr); margin-top:54px;
      padding-top:46px; }}
    .chat-intro p {{ color:#c8c7bd; max-width:430px; }}
    .chat-panel {{ background:var(--paper); box-shadow:10px 10px 0 var(--rust);
      color:var(--ink); min-width:0; padding:22px; }}
    .messages {{ display:flex; flex-direction:column; gap:12px; max-height:390px;
      min-height:230px; overflow:auto; padding:4px 4px 18px; }}
    .message {{ border:1px solid var(--ink); line-height:1.5; max-width:88%;
      padding:12px 14px; white-space:pre-wrap; }}
    .message.assistant {{ align-self:flex-start; background:#fffdf4; }}
    .message.user {{ align-self:flex-end; background:var(--acid); }}
    form {{ border-top:1px solid var(--ink); display:grid; gap:10px;
      grid-template-columns:1fr auto; padding-top:16px; }}
    textarea {{ background:#fffdf4; border:1px solid var(--ink); border-radius:0;
      color:var(--ink); font:inherit; min-height:82px; padding:12px; resize:vertical; }}
    button {{ align-self:stretch; background:var(--ink); border:1px solid var(--ink);
      border-radius:0; color:var(--paper); cursor:pointer; font:900 12px inherit;
      letter-spacing:.08em; padding:0 20px; text-transform:uppercase; }}
    button:disabled {{ cursor:wait; opacity:.55; }}
    #chat-status {{ font-size:12px; margin:10px 0 0; min-height:18px; }}
    .disclosure {{ border-left:3px solid var(--acid); color:#c8c7bd; font-size:13px;
      line-height:1.5; margin-top:20px; padding-left:12px; }}
    @media(max-width:820px) {{
      main {{ padding:18px 16px 54px; }}
      nav span {{ display:none; }}
      article {{ grid-template-columns:1fr; }}
      .card-stage {{ border-bottom:1px solid #4a4a41; border-right:0; }}
      .identity {{ padding:32px 26px; }}
      .challenge-section {{ grid-template-columns:1fr; padding:26px; }}
      .challenge-answers {{ grid-template-columns:1fr; }}
      .chat-section {{ grid-template-columns:1fr; }}
      form {{ grid-template-columns:1fr; }}
      button {{ min-height:50px; }}
    }}
  </style>
</head>
<body><main>
  <nav><b>Persona Capsule</b><span>Public personality · live steering</span></nav>
  <article>
    {image_html}
    <section class="identity">
      <span class="eyebrow">A live Persona Capsule</span>
      <h1>{escape(title)}</h1>
      <p>{escape(summary)}</p>
      <div class="traits">{descriptors_html}</div>
      {f"<ul>{dimensions_html}</ul>" if dimensions_html else ""}
      {voice_html}
      <a class="share" href="{escape(share_url)}" rel="noopener">Share on X</a>
      <small>Communication-style collectible · AI-generated artwork</small>
    </section>
  </article>
  <section class="challenge-section">
    <div>
      <span class="eyebrow">One-round personality test</span>
      <h2>Do you really know {escape(title)}?</h2>
      <p>
        MiniCPM will answer the same situation twice. One answer is ordinary.
        One has this capsule's live personality steering. Pick the one that feels right.
      </p>
    </div>
    <div class="challenge-panel">
      <button class="challenge-start" id="challenge-start" type="button">
        Start the challenge
      </button>
      <div class="challenge-content" id="challenge-content" hidden>
        <p class="challenge-prompt" id="challenge-prompt"></p>
        <div class="challenge-answers">
          <button class="challenge-answer" data-index="0" type="button"></button>
          <button class="challenge-answer" data-index="1" type="button"></button>
        </div>
        <p class="challenge-result" id="challenge-result"></p>
        <a class="share challenge-share" id="challenge-share" rel="noopener" hidden>
          Share my result on X
        </a>
      </div>
    </div>
  </section>
  <section class="chat-section">
    <div class="chat-intro">
      <span class="eyebrow">Talk to the capsule</span>
      <h2>Chat with {escape(title)}.</h2>
      <p>
        Your message is answered by MiniCPM with this capsule's personality
        direction applied live during inference.
      </p>
      <div class="disclosure">
        This is an AI simulation of communication style, not the person.
        The capsule's private source messages are never shown to visitors.
      </div>
    </div>
    <div class="chat-panel">
      <div class="messages" id="chat-messages" aria-live="polite">
        <div class="message assistant">
          Ask me for an opinion, an explanation, or help thinking through a decision.
        </div>
      </div>
      <form id="chat-form">
        <textarea id="chat-input" maxlength="800"
          placeholder="Ask this capsule something…" required></textarea>
        <button id="chat-submit" type="submit">Send</button>
      </form>
      <p id="chat-status"></p>
    </div>
  </section>
</main>
<script>
  const endpoint = {chat_endpoint};
  const challengeEndpoint = {challenge_endpoint};
  const challengeGuessEndpoint = {challenge_guess_endpoint};
  const capsuleUrl = {canonical_json};
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const submit = document.getElementById("chat-submit");
  const messages = document.getElementById("chat-messages");
  const status = document.getElementById("chat-status");
  const challengeStart = document.getElementById("challenge-start");
  const challengeContent = document.getElementById("challenge-content");
  const challengePrompt = document.getElementById("challenge-prompt");
  const challengeAnswers = Array.from(document.querySelectorAll(".challenge-answer"));
  const challengeResult = document.getElementById("challenge-result");
  const challengeShare = document.getElementById("challenge-share");
  let challengeId = null;

  function addMessage(kind, text) {{
    const node = document.createElement("div");
    node.className = `message ${{kind}}`;
    node.textContent = text;
    messages.appendChild(node);
    messages.scrollTop = messages.scrollHeight;
  }}

  challengeStart.addEventListener("click", async () => {{
    if (challengeStart.disabled) return;
    challengeStart.disabled = true;
    challengeStart.textContent = "Generating two answers…";
    challengeResult.textContent = "";
    challengeShare.hidden = true;
    try {{
      const response = await fetch(challengeEndpoint, {{method: "POST"}});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The challenge could not start.");
      challengeId = payload.challenge_id;
      challengePrompt.textContent = payload.prompt;
      challengeAnswers.forEach((button, index) => {{
        button.textContent = payload.answers[index];
        button.disabled = false;
        button.classList.remove("correct", "wrong");
      }});
      challengeContent.hidden = false;
      challengeStart.textContent = "New round";
    }} catch (error) {{
      challengeResult.textContent = error.message || "The challenge could not start.";
      challengeContent.hidden = false;
      challengeStart.textContent = "Try again";
    }} finally {{
      challengeStart.disabled = false;
    }}
  }});

  challengeAnswers.forEach((button) => {{
    button.addEventListener("click", async () => {{
      if (challengeId === null) return;
      const chosenIndex = Number(button.dataset.index);
      const activeChallengeId = challengeId;
      challengeId = null;
      challengeAnswers.forEach((answer) => {{ answer.disabled = true; }});
      challengeResult.textContent = "Checking your answer…";
      try {{
        const response = await fetch(challengeGuessEndpoint, {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{
            challenge_id: activeChallengeId,
            guess: chosenIndex
          }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "The answer could not be checked.");
        challengeAnswers.forEach((answer, index) => {{
          answer.classList.add(index === payload.steered_index ? "correct" : "wrong");
        }});
        challengeResult.textContent = payload.correct
          ? "You got it. That was the live-steered capsule."
          : "Not this time. The green answer was the live-steered capsule.";
        const shareText = payload.correct
          ? "I recognized {escape(title)}'s Persona Capsule. Do you really know them?"
          : "A Persona Capsule fooled me. Can you recognize {escape(title)}'s communication style?";
        challengeShare.href =
          `https://x.com/intent/post?text=${{encodeURIComponent(shareText)}}&url=${{
            encodeURIComponent(capsuleUrl)
          }}`;
        challengeShare.hidden = false;
      }} catch (error) {{
        challengeResult.textContent = error.message || "The answer could not be checked.";
      }}
    }});
  }});

  form.addEventListener("submit", async (event) => {{
    event.preventDefault();
    const message = input.value.trim();
    if (!message || submit.disabled) return;
    addMessage("user", message);
    input.value = "";
    submit.disabled = true;
    status.textContent = "Deriving the live personality direction…";
    try {{
      const response = await fetch(endpoint, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{message}})
      }});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The capsule could not respond.");
      addMessage("assistant", payload.reply);
      status.textContent = "AI-generated · live activation steering removed after response";
    }} catch (error) {{
      addMessage("assistant", error.message || "The capsule could not respond.");
      status.textContent = "Please try again shortly.";
    }} finally {{
      submit.disabled = false;
      input.focus();
    }}
  }});
</script>
</body></html>"""
