"""Custom Gradio interface for Persona Capsule."""

from html import escape

import gradio as gr

from persona_capsule.config import Settings
from persona_capsule.demo import DEMO_CAPSULE, demo_reply
from persona_capsule.identity import IdentityGateway, Principal
from persona_capsule.library import CapsuleLibrary

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
) -> gr.Blocks:
    """Build the public app shell and deterministic demo path."""

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

        if settings.oauth_ui_available:

            def load_private_library(
                profile: gr.OAuthProfile | None,
            ) -> tuple[str, str]:
                principal = identity_gateway.resolve(profile)
                return _account_html(principal), _library_html(principal, capsule_library)

            demo.load(load_private_library, inputs=None, outputs=[account, library])
        else:

            def load_local_library() -> tuple[str, str]:
                principal = identity_gateway.resolve_local()
                return _account_html(principal), _library_html(principal, capsule_library)

            demo.load(load_local_library, inputs=None, outputs=[account, library])

    return demo
