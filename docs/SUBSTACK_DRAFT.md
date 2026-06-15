# I Tried to Turn a Writing Style Into a Portable AI Object

## Building Persona Capsule with activation steering, small open-weight models, and a lot of lessons from the model's hidden states

*Built for the Hugging Face and Gradio Build Small Hackathon 2026.*

Most AI personality features begin and end with a prompt.

You write something like "be warm, concise, and slightly playful," place it in
a system message, and hope the model keeps following it. That can be useful, but
the personality is still an instruction sitting outside the model.

I wanted to try a different question:

**Could a person's communication style become a user-controlled object that
temporarily changes how a small language model processes a response?**

That question became **Persona Capsule**.

A capsule starts with a small set of messages that the creator owns and
explicitly approves. It becomes an editable communication-style profile, a live
activation-steering recipe for MiniCPM, an anime-style collectible, an optional
synthetic voice, and a public page that other people can actually chat with.

It is not a psychological diagnosis or a digital copy of a human. It is a
deliberately limited simulation of communication style.

## The Video That Started the Technical Journey

While exploring mechanistic interpretability, I watched Hugging Face's video
[Steering LLM Behavior Without Fine-Tuning](https://youtu.be/F2jd5WuT-zg),
created by David Louapre.

The video explains a third path between prompt engineering and fine-tuning:
intervening directly in a model's activations while it generates.

A transformer passes a high-dimensional hidden state from one layer to the
next. Research suggests that many concepts and behaviors are represented as
directions in these activation spaces. If we can identify a useful direction,
we can add a scaled version of it during a forward pass:

```text
steered hidden state = original hidden state + strength × direction
```

The model weights remain unchanged. Remove the intervention and the original
model is back.

The idea is closely connected to why mechanistic interpretability is so
interesting to me. Instead of treating a language model only as an input/output
API, we can ask what is happening inside it, identify useful internal
representations, test causal interventions, and turn those experiments into
experiences people can inspect.

The Hugging Face video demonstrates this with an Eiffel Tower concept. Persona
Capsule asks whether the same broad technique can represent something more
personal and subtle: the recurring shape of how somebody communicates.

## From Contrast Pairs to a Personality Direction

The core method in Persona Capsule is contrastive activation steering.

For every approved style example, the system creates a neutral counterpart that
tries to preserve the meaning while removing the distinctive delivery. A
Shakespeare-like source might contain archaic vocabulary, theatrical rhythm and
elaborate metaphor. Its contrast should express the same idea plainly.

At a selected MiniCPM layer `l`, we measure the token activations for both:

```text
difference_l =
    mean_tokens(style activation_l)
    - mean_tokens(neutral activation_l)
```

We repeat this across the creator's approved pairs and average the differences:

```text
v_l = mean_pairs(difference_l)
```

`v_l` is the capsule's measured steering direction at that layer.

During generation, a temporary forward hook applies:

```text
h'_l = h_l + alpha × v_l
```

`alpha` controls the intervention strength. Persona Capsule normalizes and caps
the measured direction because stronger is not automatically better.
Over-steering can replace a recognizable style with repetition or incoherence.

Our current MiniCPM recipe works across layers 8, 12, 16, 20, and 24. These are
distributed through the model rather than concentrated only at the output. The
system derives the directions from the approved pairs during inference, applies
them for that generation, and removes every hook afterward.

The product always shows an unsteered baseline beside the steered answer. Both
use the same MiniCPM model and prompt. That comparison is important: people
should be able to see whether the intervention had an effect instead of being
asked to trust the word "personalized."

## The Product Loop

The complete flow is:

1. Sign in with Hugging Face.
2. Provide messages you own and confirm permission to process them.
3. Review automatic redactions and the extracted communication-style profile.
4. Edit the profile and approve only the examples the capsule may retain.
5. Compare ordinary and live-steered MiniCPM answers.
6. Generate a profile-derived anime collectible.
7. Optionally create a consented synthetic voice.
8. Fuse compatible capsules or run a blinded personality battle.
9. Preview exactly what will become public.
10. Publish a stable page with live chat and a recognition challenge.

The public challenge is called **"Do you really know me?"**

MiniCPM answers the same situation twice. One answer is ordinary and one uses
the capsule's live activation direction. The visitor has to identify which one
feels more like the creator.

This turns the technical claim into a small social game. It also provides a
more honest test than simply labeling one response "personalized."

## Making the Capsule Tangible

An activation recipe is technically interesting, but it does not feel like an
object. The rest of the system gives the capsule a form.

### Visual identity with FLUX

Black Forest Labs' FLUX.2 Klein 4B runs with an anime-style LoRA on Modal.
Approved descriptors influence expression, posture, energy, palette, geometry,
and symbolic motifs. Each capsule also receives a stable visual signature so
two people with similar high-level traits do not automatically receive the same
picture.

The visual is intentionally fictional. Persona Capsule does not attempt to
reconstruct the creator's physical likeness, and it never sends private source
messages to the image model.

### Optional voice with VoxCPM2

We originally explored a closed voice-cloning API, but the final system uses
OpenBMB's VoxCPM2 2B on Modal.

Voice is optional and consent-gated. The creator must confirm ownership or
permission, the private reference has a retention window, and the generated
sample can be included or excluded from the public projection independently.
Deletion failures are recorded as retryable cleanup work rather than being
silently treated as success.

### Blinded battles with Nemotron

Two compatible capsules can be fused by weighting their live steering
directions. For evaluation, MiniCPM generates anonymized candidates and NVIDIA
Nemotron 3 Nano 4B judges both A/B and B/A orderings.

Running both orders does not make an LLM judge objective, but it helps expose
simple position bias and makes the result more inspectable.

## The Architecture

The user experience is a custom Gradio application hosted in the official Build
Small organization on Hugging Face Spaces.

The Space handles authentication, consent, workflow logic, public pages and
durable private records. GPU-heavy work is dispatched to separate Modal
runtimes:

- MiniCPM4.1-8B for activation extraction and steered generation;
- FLUX.2 Klein 4B plus an anime LoRA for visual cards;
- OpenBMB VoxCPM2 2B for consented synthetic speech;
- NVIDIA Nemotron 3 Nano 4B for blinded battle evaluation.

Hugging Face OAuth binds private capsules to their creators. A private Hugging
Face Dataset repository provides durable storage across Space rebuilds.
FastAPI serves the public image, audio, chat, challenge and metadata routes.

## Privacy Was an Architecture Requirement

Personal messages and voice cannot be treated like ordinary demo inputs.
Persona Capsule follows a few rules:

- Creating a capsule is private by default.
- Obvious sensitive information is redacted before analysis.
- The creator reviews and approves the retained examples.
- Permanent user activation tensors are not stored.
- Private capsule records and public projections are separate objects.
- Public pages never reveal the source examples.
- Creators can unpublish without deleting their private capsule.
- Provider failures cannot silently erase or corrupt a capsule.

The capsule stores the recipe, approved contrasts, model version and
compatibility metadata. The actual steering directions are temporary runtime
data.

## What Broke Along the Way

The most useful part of building this was discovering how easy it is to produce
a steering demo that looks convincing for the wrong reason.

We encountered zero-magnitude directions when a layer did not capture a useful
difference. Early contrast pairs sometimes preserved too much of the source
style, so the vector learned formatting artifacts instead of the intended
signal. Applying a weak intervention only to the newest token made literary
style nearly invisible. Applying too much produced repetition.

MiniCPM also began answering English prompts in Chinese. Because the baseline
did the same thing, this was not caused by the steering vector; the generation
template needed a stronger language instruction.

Outside the model, Hugging Face OAuth redirects required careful separation
between creator and visitor routes. Generated public images initially vanished
after a Space rebuild because they lived only on ephemeral disk. They now use a
durable artifact boundary.

These failures changed the implementation:

- clearer neutral contrasts;
- multiple selected layers;
- bounded vector magnitudes;
- consistent language instructions;
- request-scoped hook cleanup diagnostics;
- persistent public artifacts;
- a public route that includes chat, not only an image.

## Building With Codex

OpenAI Codex was used throughout the project as an attributed engineering
collaborator.

It helped audit the initial plan against provider documentation, turn the plan
into a PRD and GitHub issue slices, design the architecture, implement the
services and UI, write tests, inspect the browser, diagnose provider failures,
and deploy the Modal runtimes and Hugging Face Space.

Codex-authored commits contain:

```text
Co-authored-by: Codex <noreply@openai.com>
```

That public history matters for the Build Small OpenAI track, but it also
reflects how the project was actually built: through an iterative collaboration
that moved between product decisions, model behavior, infrastructure and user
experience.

## What I Want to Explore Next

Persona Capsule currently uses straightforward contrastive activation
differences. There is much more to investigate:

- measuring which layers carry stable style features;
- comparing additive steering with clamping;
- evaluating whether directions generalize across prompt categories;
- separating surface vocabulary from deeper rhetorical structure;
- using sparse autoencoders to identify interpretable features;
- testing whether a capsule can explain which internal features influenced a
  response;
- designing better human evaluations for "recognizable style."

The larger motivation is mechanistic interpretability: learning enough about a
model's internal representations to make interventions that are causal,
bounded, testable and honest about their limitations.

## Try It

Persona Capsule is live in the official Build Small Hackathon organization:

- [Hugging Face Space](https://huggingface.co/spaces/build-small-hackathon/persona-capsule)
- [GitHub repository](https://github.com/perfect7613/persona-capsule)
- [Activation-steering inspiration](https://youtu.be/F2jd5WuT-zg)

Create a capsule, compare the baseline and steered outputs, and share the
recognition challenge with somebody who knows how you write.

Then ask them the question at the center of the project:

**Do you really know me?**

---

Thanks to the Hugging Face and Gradio teams, the Build Small organizers, and
the OpenBMB, OpenAI, NVIDIA, Modal, Black Forest Labs, JetBrains and Cohere Labs
partner teams for supporting the hackathon.
