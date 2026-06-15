# Persona Capsule Demo Production Plan

## Deliverables

Produce these from one clean recording session:

1. **Judge demo:** 2 minutes 45 seconds, 16:9, 1080p.
2. **Hero montage:** 8-12 seconds for the first X post.
3. **Steering clip:** 15-20 seconds for the activation-steering posts.
4. **Art clip:** 8-12 seconds showing the FLUX card reveal.
5. **Voice clip:** 10-15 seconds showing consent and VoxCPM2 playback.
6. **Battle clip:** 12-18 seconds showing fusion and Nemotron judging.
7. **Share clip:** 15-20 seconds showing the public chat and recognition game.

Do not make the thread depend on the full demo video. Attach the short clip that
proves each claim directly to the relevant post.

## Preparation

- Record at 1920x1080, 30 fps.
- Use a clean browser profile at 100% zoom and hide bookmarks.
- Pre-warm every Modal runtime immediately before recording.
- Use a synthetic Shakespeare-style capsule, not private messages.
- Prepare four strong approved style/neutral pairs.
- Generate the image and voice once before the session to confirm providers.
- Keep a published capsule URL ready for the visitor sequence.
- Disable desktop notifications and hide terminal windows containing secrets.
- Use cursor emphasis only for important clicks.
- Add captions; many X viewers watch without sound.

Never show `.env`, provider dashboards, access tokens, private datasets, raw
voice uploads or identifying source messages.

## Main Film

### 0:00-0:12 — Hook

**Picture**

Fast sequence: paste messages, compare two outputs, reveal the anime card, open
the public challenge.

**Narration**

> What if the way you communicate could become a private AI object that your
> friends can see, hear and talk to?

**On-screen text**

`PERSONA CAPSULE`

`Your communication style, live inside a small model`

### 0:12-0:33 — Create With Consent

**Picture**

Show Hugging Face sign-in, ownership confirmation, synthetic source messages,
redaction and the editable style profile.

**Narration**

> Persona Capsule starts with messages you own. It redacts obvious sensitive
> information, proposes a communication-style profile, and keeps nothing until
> you review and approve it.

**Proof to keep visible**

- consent checkbox;
- redaction result;
- editable profile;
- approved-example count.

### 0:33-1:12 — The Main Technical Moment

**Picture**

Approve the capsule and show the success notification opening the Test step.
Enter one practical prompt. Hold on the baseline and steered outputs long enough
to read the difference. Briefly show the diagnostics.

**Narration**

> The central experiment is activation steering on OpenBMB's MiniCPM4.1-8B.
> For every approved example, we compare the model's internal activations with
> a neutral version carrying the same meaning. Their average difference becomes
> a temporary direction for this capsule.
>
> During generation, hooks add that direction inside selected middle layers.
> The weights never change. The strength is bounded, and every hook is removed
> after the response. The same prompt also runs without steering, so the effect
> is visible rather than assumed.

**On-screen formula**

`v_l = mean(style activations - neutral activations)`

`h'_l = h_l + alpha * v_l`

**Proof to keep visible**

- same prompt for both answers;
- recognizable output difference;
- model and recipe;
- `Hooks active after request: False`.

### 1:12-1:31 — Make It Tangible

**Picture**

Generate the anime card and show the final image full-screen.

**Narration**

> Black Forest Labs' FLUX.2 Klein with an anime LoRA turns the approved profile
> into a collectible visual identity. Traits shape posture, expression, palette
> and motifs, while private messages never go to the image model.

### 1:31-1:47 — Give It A Voice

**Picture**

Show the explicit voice-permission control, generate a short VoxCPM2 sample and
play two seconds of clean audio.

**Narration**

> An optional, consented OpenBMB VoxCPM2 reference gives the capsule synthetic
> speech. The reference remains private, has a retention policy and can be
> deleted independently.

### 1:47-2:08 — Fusion And Evaluation

**Picture**

Fuse two compatible capsules with a visible weight slider. Immediately run the
blinded battle and reveal the Nemotron result.

**Narration**

> Compatible capsules can be mixed without merging permanent tensors. For a
> playful evaluation, MiniCPM creates anonymous candidates and NVIDIA Nemotron
> judges both A/B and B/A orderings to reduce position bias.

### 2:08-2:35 — Share A Living Capsule

**Picture**

Preview the public projection, publish, copy the link, then open it in a private
browser window. Send one chat message and play "Do you really know me?" Reveal
the steered answer.

**Narration**

> Publishing never exposes the private source. It creates a reviewed public
> projection with its own card, live-steered chat and a recognition game: can a
> friend distinguish the capsule from the ordinary model?

### 2:35-2:47 — Architecture And Credits

**Picture**

Show a clean architecture card:

`Gradio Space -> Modal -> MiniCPM / FLUX / VoxCPM2 / Nemotron`

Then show the Git history with a Codex co-author trailer.

**Narration**

> Hugging Face and Gradio host the experience, Modal runs the GPU workloads,
> and OpenAI Codex helped audit, design, implement, test and deploy the project
> with public commit attribution.

### 2:47-2:55 — Close

**Picture**

Final anime card beside the public challenge.

**Narration**

> A personality is more than a prompt. Try Persona Capsule and see whether the
> model's temporary internal direction feels recognizable.

**On-screen text**

`DO YOU REALLY KNOW ME?`

`huggingface.co/spaces/build-small-hackathon/persona-capsule`

## Short Clip Exports

| Clip | Start/end source | X usage | Caption |
| --- | --- | --- | --- |
| Hero | 0:00-0:12 | Thread 1 | `Messages become a live Persona Capsule` |
| Inspiration | custom 8 sec | Thread 2 | `From mechanistic interpretability to a product` |
| Math | 0:37-0:54 | Threads 3-5 | `No weight update. A temporary activation direction.` |
| Comparison | 0:54-1:12 | Thread 6 | `Same model + same prompt; steering is the variable` |
| Architecture | custom 12 sec | Thread 7 | `Open-weight models, dispatched on Modal` |
| Art | 1:12-1:31 | Thread 8 | `A profile-derived FLUX collectible` |
| Voice | 1:31-1:47 | Thread 9 | `Optional, consented VoxCPM2 voice` |
| Battle | 1:47-2:08 | Thread 10 | `Nemotron judges both candidate orders` |
| Share | 2:08-2:35 | Thread 12 | `The shared link is a chat, not only an image` |
| Codex | 2:35-2:47 | Thread 13 | `Attributed engineering collaboration` |

## Editing Notes

- Remove all model-loading waits; replace them with a short branded progress
  card saying `Deriving live direction on Modal`.
- Use hard cuts inside a workflow and restrained wipes between major sections.
- Keep UI audio muted except for the VoxCPM2 sample.
- Put model names in lower thirds when first introduced.
- Do not call the steered output "fine-tuned."
- Do not imply that a capsule recreates or knows the complete person.
- Use the phrase `communication-style simulation` at least once.
- End with the direct Space URL, not only the GitHub repository.

## Final Verification

- Verify every feature shown works in the deployed Space.
- Verify the public link in the recording opens without creator access.
- Watch once with sound off to confirm captions carry the story.
- Check that all text is readable on a phone-sized player.
- Upload the final film publicly and replace `[DEMO_URL]` in the social thread,
  README and Space README.
