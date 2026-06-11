# Persona Capsule Plan Audit

Reviewed on June 11, 2026 against the supplied Word document and the current
official hackathon, model, library, and infrastructure documentation.

## Executive Verdict

Persona Capsule is a strong hackathon concept because it has:

- a clear visual object: the capsule card;
- an understandable interaction: create, blend, battle, share;
- a technically interesting core: activation steering;
- natural demo moments;
- plausible sponsor alignment with MiniCPM, Nemotron, Modal, Codex, and FLUX.

The current plan is not buildable as written. It combines a good product pitch
with several untested or false technical assumptions, and it scopes roughly
two to four weeks of work into the remaining four calendar days.

The strongest version is not "five models and every badge." It is:

1. one credible persona-steering proof;
2. one polished card-generation path;
3. one memorable social interaction, probably fusion or battle;
4. an honest evaluation showing what works and what does not.

## Severity Summary

### Critical

1. The document says "Hackathon 2025." The current event deadline is June 15,
   2026. Today is June 11, 2026, so the original seven-day schedule is already
   obsolete.
2. The steering-vector sample code does not match the library API. The library
   has no vector arithmetic operators and no `.save()` method.
3. A steering vector is model-specific. It cannot be injected into "any small
   model" because its tensor dimensions and layer indices depend on the base
   model.
4. The plan never defines a defensible algorithm for turning 20 unlabelled chat
   messages into five OCEAN contrastive vector datasets.
5. The user flow promises a capsule in about 60 seconds, while the plan also
   includes a five-minute writing LoRA, a one-hour image LoRA, and a voice clone.
6. Voice cloning cannot be derived from text messages. It requires recorded
   audio and explicit consent.
7. The default Hugging Face Space disk is 50 GB. The listed inference artifacts
   total more than that before Python packages and caches.
8. The plan claims Tiny Titan eligibility while using an 8.185B MiniCPM model.
   Tiny Titan requires models at or below 4B.

### High

1. Simple `0.6 * vector_a + 0.4 * vector_b` does not guarantee a perceptual
   60/40 blend. Vector norms and layer effects must be normalized and calibrated.
2. The claimed "70-90% of fine-tuning quality" is not supported by the sources
   reviewed and needs an experiment or removal.
3. FLUX LoRA ownership is unclear. A global card-style LoRA is feasible. A
   per-user LoRA is not compatible with a one-minute creation flow.
4. The MiniCPM LoRA ownership is also unclear. If it is per user, every capsule
   needs a training job. If it is global, it does not capture the individual.
5. A single Nemotron score is not a reliable fidelity metric because LLM judges
   exhibit order, verbosity, prompt-injection, and style biases.
6. Publishing capsules automatically to the Hub needs authentication, namespace
   ownership, privacy choices, deletion, and abuse handling.
7. Shared `.pt` files are a poor interchange format. They are model-specific and
   common PyTorch loading paths can deserialize unsafe objects.
8. Chat exports may contain other people's messages and sensitive information.
   The plan has no consent, redaction, retention, or deletion policy.

## What The Product Actually Is

The plan currently mixes three different product definitions:

1. **Psychological persona extraction**
   Claims to infer OCEAN personality from messages.
2. **Writing-style imitation**
   Attempts to reproduce vocabulary, punctuation, cadence, and tone.
3. **Portable runtime behavior controls**
   Packages activation vectors and structured metadata.

These are related but not equivalent.

For a hackathon, call the result a **communication-style capsule**, not a
psychological profile. OCEAN can remain a playful control surface, but it should
be user-adjustable and described as an inferred style lens, not a diagnosis.

## Core Steering Design

### What the library supports

`steering-vectors` 0.12.2:

- supports decoder-only PyTorch/Hugging Face models;
- extracts contrastive activation differences;
- can infer conventional decoder layer names;
- accepts selected positive and negative prompt pairs;
- applies vectors with a runtime multiplier;
- supports Transformers from 4.35.2 up to, but not including, 5.0.

MiniCPM4.1-8B declares Transformers 4.56.1 and uses conventional
`model.layers.<n>` decoder names. This makes compatibility plausible, not
confirmed. The library test suite does not test MiniCPM.

### What the library does not support

The following code in the plan is pseudocode, not working API:

```python
v_you * 0.6 + v_hemingway * 0.4
sv.save("capsule.pt")
```

You must implement:

- layer-by-layer vector normalization;
- compatibility checks for model ID, revision, hidden size, layer set, and
  layer type;
- weighted composition;
- safe serialization, preferably `safetensors` plus JSON metadata;
- calibration of multipliers;
- versioning and integrity hashes.

### The missing extraction algorithm

Twenty chat messages are not contrastive pairs. The plan needs to choose one of
these designs:

#### Design A: Global OCEAN basis, user-specific coefficients

1. Build five positive/negative trait datasets once.
2. Extract five basis vectors once for MiniCPM.
3. Estimate the user's coefficients from their messages.
4. Apply the weighted basis at generation time.

This is the most feasible design. It creates portable coefficients, while the
actual vectors remain tied to MiniCPM.

#### Design B: User-specific residual vector

1. Generate neutral paraphrases of each user message.
2. Treat the original as positive and the neutral paraphrase as negative.
3. Extract a user-style residual vector.

This is closer to "your voice," but the quality depends heavily on the
paraphraser. It also captures topic and content leakage unless prompts are
carefully controlled.

#### Design C: User-specific OCEAN vectors

This requires generating or labelling positive and negative examples for every
trait and user. It is the least defensible option with only 20 messages and the
least likely to fit the deadline.

**Recommendation:** implement A first, then add B only if a spike proves it.

### Fusion requirements

Two capsules can be blended only when they share:

- the exact base model and revision;
- hidden size;
- layer type;
- selected layer indices;
- extraction method;
- normalization method.

A better capsule stores coefficients and metadata:

```json
{
  "format_version": "0.1",
  "base_model": "openbmb/MiniCPM4.1-8B",
  "model_revision": "...",
  "vector_basis": "persona-capsule/ocean-minicpm41-v1",
  "coefficients": {
    "openness": 0.64,
    "conscientiousness": 0.21,
    "extraversion": -0.18,
    "agreeableness": 0.35,
    "neuroticism": -0.12
  }
}
```

This is genuinely composable. The large basis vectors are downloaded once.

## LoRA Strategy

### MiniCPM writing LoRA

Rank 4, `q_proj` only, one epoch, and 20 messages is an experiment, not a
validated recipe. It may learn almost nothing or memorize wording. It also
complicates steering because vectors extracted from the base model may behave
differently after adapter weights are applied.

For the deadline:

- do not train a writing LoRA per user;
- optionally train one documented demo adapter to qualify for Well-Tuned;
- do not claim that adapter is the user's capsule unless evaluation proves it;
- extract or validate steering vectors with the adapter enabled.

### FLUX.2 Klein LoRA

The official BFL article recommends roughly 15-40 images, content-only captions,
a rare trigger token, training the base model, and using checkpoints rather than
assuming the final checkpoint is best.

The plan's "27 is optimal," ">40 overfits," and "60-75% is the best checkpoint"
are stronger claims than the source supports.

Use one global **Persona Capsule card-art LoRA** trained before the demo. A user
profile then controls the prompt, palette, symbols, and composition. Do not
train a FLUX LoRA during capsule creation.

## Voice

The flow accepts text, but voice cloning requires audio. ElevenLabs Instant
Voice Cloning is not on the free plan; it starts on the Starter tier, currently
listed at $6/month.

Choose one:

1. Remove voice from the MVP.
2. Add an optional, explicit audio-upload step with consent.
3. Use a designed synthetic voice rather than calling it a clone.
4. Investigate OpenBMB's VoxCPM2 for stronger sponsor alignment and a local path.

Recommendation: remove voice from the judged core unless the text and card paths
are already working.

## Nemotron Battle And Evaluation

Nemotron 3 Nano 4B is 3.973B parameters and fits the event cap. Its model card
uses a custom Nemotron-H hybrid architecture. It is appropriate as a judge, but
"reasoning-on mode out of the box" needs an implementation check against the
4B model's chat template; the linked NVIDIA usage guide demonstrates the 30B
model through OpenRouter, not this local 4B checkpoint.

A defensible battle:

1. Use the same held-out challenge for both capsules.
2. Hide capsule identities from the judge.
3. Evaluate against each creator's short style rubric and examples.
4. Run both A/B and B/A ordering.
5. Score multiple dimensions: style match, content quality, and safety.
6. Treat the result as game feedback, not scientific truth.
7. Escape user-controlled text and defend the judge prompt from injection.

Do not print one "87/100 fidelity" score unless it comes from a repeatable
evaluation with held-out examples.

## Multilingual

The document points to the older `aya-expanse-3b`, while the 2026 event promotes
the Tiny Aya family. `tiny-aya-global` is about 3.349B parameters and covers a
large language set, but is gated and licensed CC-BY-NC-4.0.

Activation steering does not automatically transfer across languages merely
because hooks operate on Transformer layers. Cross-lingual effect is an
experiment. The app needs at least a small language matrix with measured results.

Recommendation: remove Tiny Aya from the core architecture. Add one multilingual
demo only after the English path is stable.

## Deployment And Storage

### ZeroGPU

Current ZeroGPU documentation describes:

- 70 GB RAM and 48 GB VRAM for a Large allocation;
- 141 GB RAM and 96 GB VRAM for X-Large;
- about 30 minutes of daily free quota and about 25 minutes per call;
- PRO grants five times more daily quota;
- only Gradio SDK Spaces are supported.

The plan's exact "40 minutes/day" and "24 sessions/day" figures are not in the
current official documentation and should be removed.

### Memory

The plan counts model weights but not:

- CUDA/runtime overhead;
- KV cache;
- activations used during steering extraction;
- duplicated tensors during loading;
- image pipeline components;
- concurrent requests.

Three BF16 LLMs totalling about 31 GB may fit in 48 GB for light inference, but
it is not safe to promise. Adding FLUX in the same process is especially risky.

### Disk

Approximate repository storage:

- MiniCPM4.1-8B: 16.4 GB;
- Nemotron 4B: 10.8 GB;
- Tiny Aya Global: 6.7 GB;
- FLUX.2 Klein 4B: 23.7 GB.

That is about 57.6 GB before packages, generated assets, and caches. The default
Space disk is 50 GB and is not persistent.

Use fewer models, quantized artifacts, separate services, or paid persistent
storage. Define where cards, metadata, and generated images live.

### gr.Server and sharing

`gr.Server` is a valid route to a custom frontend plus a FastAPI backend. It can
support path-specific HTML routes for social metadata, but those routes and
asset storage must be designed explicitly. A static Gradio `head` string alone
will not generate different Open Graph metadata for every capsule.

## Modal

The proposed Modal pattern is technically sound:

- `modal.App`;
- GPU functions;
- named `modal.Volume` mounts for checkpoints;
- `modal.Secret` for Hugging Face tokens;
- explicit timeouts;
- A100/A100-80GB training.

Current listed prices make the rough estimate closer to:

- five minutes on A100: about $0.18;
- four hours on A100-80GB: about $9.99;
- five hours on A10: about $5.51;
- total GPU spend: about $15.68, plus CPU, memory, storage, and retries.

The budget remains comfortable under $250, but "$14.20 total" is too precise
before the training scripts have been benchmarked.

## Privacy, Safety, And Ownership

The plan needs a minimal trust design:

- process only messages the user owns or has permission to use;
- explain that chat exports can include other people's text;
- redact phone numbers, emails, URLs, and named entities by default;
- do not publish anything without a separate explicit opt-in;
- make generated capsules private by default;
- define retention and deletion;
- do not store raw chat after extraction unless needed;
- require consent for any audio upload;
- prevent impersonation claims and clearly label generated speech;
- provide a report/delete mechanism for public capsules.

Avoid automatic psychological labels. Let users edit the inferred profile before
generation and sharing.

## Portability

PersonaSpec/PIF is a 2026 draft with a small, early repository. PIF is useful as
a structured JSON export, but it is not evidence that many platforms can consume
the capsule.

Split exports:

1. `.persona`: human-readable profile, examples, permissions, and metadata.
2. `.safetensors`: model-specific activation vectors, if needed.
3. `manifest.json`: hashes, model revision, layer mapping, license, and versions.

Never describe the tensors as compatible with any decoder-only model.

## Prize And Badge Corrections

### Valid sponsor prizes

The current field guide lists dedicated sponsor prizes for:

- OpenBMB;
- OpenAI Codex;
- NVIDIA;
- Modal.

Black Forest Labs and Cohere are supporting partners, but the current prize
table does not list separate "BFL Contribution" or "Cohere Contribution" cash
awards. Remove those assumed prize rows unless organizers confirm otherwise.

### Tiny Titan

The field guide says models must be at or below 4B. MiniCPM4.1-8B is
8,185,253,888 parameters. The current design does not qualify.

Using one 0.5B side model does not make an app Tiny Titan while the core uses 8B.

### Off the Grid

"No cloud APIs; the whole thing runs on the model in front of you." A project
whose core flow requires ZeroGPU, Modal, ElevenLabs, or hosted APIs does not
honestly satisfy this achievement.

### Well-Tuned

The app must use a fine-tuned model that is published on Hugging Face. Publishing
an unused adapter is not enough.

### Llama Champion

A last-day GGUF smoke test is weak evidence. The submitted app should expose a
real llama.cpp path or provide a reproducible local mode.

### OpenAI prize

The official materials are internally inconsistent: the current field-guide
source lists $5k/$3k/$1k while also claiming a $10k total, and the main event
page lists $5k/$3k/$2k. Treat the event page as the current public figure and
ask organizers if prize planning depends on it.

## Four-Day Scope Recommendation

### Must ship

1. MiniCPM text generation with one validated steering path.
2. A user-editable style profile derived from pasted messages.
3. Capsule card UI using `gr.Server`.
4. Fusion between two compatible profiles.
5. One global FLUX card-art LoRA trained on Modal.
6. Save/load a safe capsule manifest.
7. Demo video, social post, README, tags, and attribution.

### Ship only if the core is stable

1. Nemotron battle with order-swapped judging.
2. PIF export.
3. Public share links.
4. One multilingual example.

### Remove from this submission

1. Per-user MiniCPM LoRA.
2. Per-user FLUX LoRA.
3. ElevenLabs voice cloning.
4. Automatic Hub publishing.
5. "Any decoder-only model" compatibility.
6. All-badges strategy.
7. Five simultaneously loaded models.

## Proposed Build Sequence

### June 11

- prove MiniCPM plus `steering-vectors` on one trait and three layers;
- measure latency and memory;
- implement safe save/load and vector composition;
- decide whether the extraction is global-basis or residual.

### June 12

- build profile inference and editable controls;
- implement capsule creation and fusion;
- create the custom card UI shell.

### June 13

- train and select the global FLUX LoRA on Modal;
- wire card-image generation;
- add storage for generated cards.

### June 14

- add battle only if the core is reliable;
- deploy the Space;
- test cold starts, quota behavior, error states, and mobile layout;
- write the README and evaluation notes.

### June 15

- record the demo early;
- publish social post;
- add exact submission tags and links;
- freeze features and fix only submission-blocking defects.

## Go/No-Go Experiments

Run these before building the full UI:

1. MiniCPM loads with the exact Transformers version and `trust_remote_code`.
2. The steering library identifies 32 decoder layers.
3. One contrastive vector visibly changes outputs without destroying coherence.
4. Two normalized vectors can be combined reproducibly.
5. Save/load reproduces identical vector metadata and close output behavior.
6. A full request fits ZeroGPU memory and call duration.
7. The global FLUX LoRA transfers from base training to distilled inference.

If experiment 3 fails, pivot from activation steering to prompt/profile
conditioning and present steering as an experimental comparison, not the product.

## Document Quality

The Word file renders cleanly, but:

- page 12 is mostly blank because the final references spill onto it;
- several tables are dense and use small text;
- URLs are plain text rather than actual hyperlinks;
- both images lack alt text;
- all 15 tables lack marked header rows for assistive technology;
- the title and footer still say 2025;
- many numerical claims are presented as confirmed without citations or
  experiment status.

The next revision should label claims as one of:

- verified from official documentation;
- measured in our prototype;
- design target;
- open experiment.

## Sources Reviewed

- [Build Small event page](https://huggingface.co/build-small-hackathon)
- [Build Small field guide](https://huggingface.co/spaces/build-small-hackathon/field-guide)
- [Build Small field-guide source](https://huggingface.co/spaces/build-small-hackathon/field-guide/tree/main)
- [Modal documentation](https://modal.com/docs)
- [Modal pricing](https://modal.com/pricing)
- [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Hugging Face storage](https://huggingface.co/docs/hub/spaces-storage)
- [MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)
- [NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [NVIDIA Nemotron repository](https://github.com/NVIDIA-NeMo/Nemotron)
- [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global)
- [FLUX.2 Klein LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
- [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [steering-vectors](https://github.com/steering-vectors/steering-vectors)
- [PERSONA framework](https://github.com/xcfcode/persona)
- [PersonaSpec](https://personaspec.org/)
- [Gradio custom frontend guide](https://www.gradio.app/guides/customizing-your-demo-with-html-and-js)
- [ElevenLabs pricing](https://elevenlabs.io/pricing)
- [ElevenLabs voice cloning](https://elevenlabs.io/docs/creative-platform/voices/voice-cloning)

## First Decision

Before choosing architecture, decide what winning means:

1. **Best polished product demo:** keep MiniCPM4.1-8B, drop Tiny Titan and most
   badge work, and focus on creation plus fusion.
2. **Maximum prize eligibility:** redesign around models at or below 4B and add
   a genuine local llama.cpp mode, accepting weaker generation quality and more
   integration risk.
3. **Strongest technical research story:** focus almost entirely on extraction,
   vector composition, and evaluation, with a lighter card UI.

This decision controls model selection, scope, evaluation, and badge strategy.
