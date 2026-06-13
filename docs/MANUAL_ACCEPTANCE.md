# Manual Acceptance

Complete this checklist on the deployed Space after rotated credentials are
installed. Record failures as GitHub issues instead of editing this checklist to
hide them.

## Desktop

- [ ] Public demo loads while signed out.
- [ ] Hugging Face OAuth gates creator operations.
- [ ] Creation, redaction, review, approval, and private save complete.
- [ ] Baseline and live-steered MiniCPM outputs render with diagnostics.
- [ ] FLUX card and 1200 x 628 social card render.
- [ ] Publishing preview matches the public page and unpublish works.
- [ ] Fusion enforces compatibility and stores provenance.
- [ ] Nemotron battle shows blinded, order-swapped structured scoring.
- [ ] Deep Capsule estimate, start, poll, resume, and cancel work.
- [ ] ElevenLabs cloning and deletion work when the optional provider is enabled.
- [ ] Provider kill switches and quota messages preserve the saved capsule.

## Mobile

- [ ] Core creation controls remain readable and tappable.
- [ ] Baseline/steered comparison does not overflow horizontally.
- [ ] Card preview and public share page fit the viewport.
- [ ] Fusion, battle, and Deep Capsule status controls remain usable.

## Owner Review

- [ ] No private exemplar, audio, token, or provider secret appears in telemetry.
- [ ] Demo video follows `docs/DEMO_SCRIPT.md`.
- [ ] Demo-video URL is added to the README.
- [ ] X/social post is reviewed and its URL is added to the README.
- [ ] `scripts/check_submission.py --strict` passes.
