# Persona Capsule

## Product Requirements Document

**Status:** Draft for review  
**Prepared:** June 11, 2026  
**Submission deadline:** June 15, 2026  
**Target track:** Thousand Token Wood  
**Submission host:** Hugging Face Gradio Space in `build-small-hackathon`  
**Planned source repository:** Public GitHub repository `perfect7613/persona-capsule`

## Problem Statement

People develop a recognizable communication style across messages, writing,
voice, and visual preferences, but that style is trapped inside individual
platforms and large conversation histories. Existing persona products generally
offer a prompt, static profile, or closed digital clone. They do not give users
a compact artifact they can inspect, edit, combine, export, and share.

The user needs a playful but technically credible way to turn their own messages
and voice into a portable personality artifact. The artifact should visibly
affect model behavior, have a memorable visual and spoken identity, support
composition with another capsule, and be shareable as a collectible card.

The challenge is to deliver that experience with small open-weight models while
being honest about model-specific activation steering, protecting uploaded
personal data, surviving constrained GPU resources, and meeting the Build Small
submission requirements by June 15, 2026.

## Solution

Persona Capsule is an authenticated Gradio application that converts a user's
messages, optional voice recording, and editable style profile into a
multimodal collectible capsule card.

The capsule contains:

- an editable communication-style profile;
- user-approved style exemplars and neutral contrast examples used to derive
  MiniCPM steering vectors inside each inference session;
- the exact MiniCPM steering recipe, selected layers, and calibration settings,
  without a permanently serialized user steering tensor;
- optional MiniCPM LoRA metadata for a Deep Capsule;
- FLUX-generated card art and optional visual LoRA metadata;
- an ElevenLabs Instant Voice Clone reference or pre-generated voice sample;
- provenance, consent, privacy, model-version, and compatibility metadata;
- a PersonaSpec-compatible `.persona` profile export;
- a safe capsule manifest and separately stored model artifacts.

Users can create a capsule, compare steered and baseline writing, converse with
it, hear it speak, fuse it with another compatible capsule, battle capsules
using a blinded Nemotron judge, and publish an X-compatible card URL.

The product provides two creation modes:

### Quick Capsule

Quick Capsule is the primary live demo path.

- Accept approximately 20 user-owned messages.
- Redact obvious sensitive data before model processing.
- Infer an editable communication-style profile.
- Let the user approve a small set of style exemplars and neutralized contrast
  examples.
- During live MiniCPM inference, extract a user-specific activation direction
  from those examples and immediately apply it to generation.
- Generate card art using a pre-trained global Persona Capsule FLUX LoRA.
- Optionally create an ElevenLabs Instant Voice Clone from consented audio.
- Assemble a capsule in a target of one to three minutes after models are warm.

### Deep Capsule

Deep Capsule runs asynchronously and enriches an existing Quick Capsule.

- Train a small personal MiniCPM writing LoRA on Modal.
- Optionally create synthetic personality-art training images and train a
  per-capsule FLUX LoRA on Modal.
- Evaluate multiple checkpoints.
- Retain the Quick Capsule as a usable fallback if training fails.
- Target completion in approximately 10 to 90 minutes, depending on selected
  training options.

## Goals

1. Make activation steering the central, visible product capability.
2. Derive and apply user steering during live MiniCPM inference rather than
   substituting a precomputed generic user vector.
3. Demonstrate MiniCPM4.1-8B as the primary persona generation model.
4. Use NVIDIA Nemotron 3 Nano 4B as a meaningful battle and evaluation model.
5. Use Modal for model training and Nemotron inference where appropriate.
6. Produce consistent, personality-derived FLUX.2 Klein card art.
7. Support optional ElevenLabs Instant Voice Cloning through its API.
8. Require Hugging Face authentication for creation and account management.
9. Keep capsules private until their creator explicitly publishes them.
10. Provide dynamic X/Open Graph share pages for published capsules.
11. Meet the Build Small hosting, demo, social-post, README, and tagging rules.

## Success Criteria

The submission is successful when:

1. A signed-in user can create a Quick Capsule from their own messages.
2. Steering produces a visible, repeatable style change without making the
   output unusable.
3. User-specific steering is extracted and applied within the live inference
   workflow, with no persisted user activation tensor required.
4. The application shows baseline and steered outputs side by side.
5. Two compatible capsules can be fused with calibrated weighted composition.
6. A capsule has generated card art and an optional playable voice sample.
7. Nemotron can run a blinded, order-swapped capsule battle.
8. A user can publish and unpublish a capsule.
9. A published URL renders X-compatible metadata and a 1200x628 card image.
10. Unselected raw messages and original chat exports are deleted after
    processing by default.
11. Capsule deletion removes private records, public pages, generated assets,
    retained ElevenLabs clones, and user-specific training artifacts.
12. The live app is deployed as a Gradio Space inside the official hackathon
    organization.
13. A demo video can show creation, steering, fusion, voice, battle, and sharing
    even if live GPU capacity is unavailable during judging.

## User Stories

1. As a visitor, I want to understand the product before signing in, so that I
   can decide whether I trust it with personal data.
2. As a visitor, I want to see a pre-built demonstration capsule, so that I can
   experience the concept without uploading my information.
3. As a visitor, I want to sign in with Hugging Face, so that I do not need to
   create another account.
4. As a signed-in user, I want the application to recognize my Hugging Face
   identity, so that my capsules belong only to me.
5. As a signed-in user, I want to see my existing capsules, so that I can return
   to, publish, or delete them.
6. As a user, I want a clear explanation of what a communication-style capsule
   is, so that I do not mistake it for a psychological diagnosis.
7. As a user, I want to paste approximately 20 messages, so that the application
   has examples of how I communicate.
8. As a user, I want to upload a supported chat export, so that I do not have to
   copy messages manually.
9. As a user, I want the importer to distinguish my messages from other
   participants' messages, so that the capsule represents me rather than the
   whole conversation.
10. As a user, I want to confirm that I own or have permission to process the
    submitted messages, so that other people's data is not used silently.
11. As a user, I want a preview of extracted messages before processing, so that
    I can remove private or unrepresentative examples.
12. As a user, I want automatic redaction of obvious emails, phone numbers,
    links, and identifiers, so that accidental sensitive data is minimized.
13. As a user, I want to edit or disable proposed redactions, so that useful
    writing patterns are not destroyed without my knowledge.
14. As a user, I want feedback when my sample is too short, repetitive, or
    dominated by copied text, so that I can provide better input.
15. As a user, I want to choose Quick Capsule, so that I can get a result during
    a live session.
16. As a user, I want to choose Deep Capsule after Quick creation, so that I can
    request higher personalization without blocking the initial experience.
17. As a user, I want to see processing stages and realistic status messages, so
    that I understand what the application is doing.
18. As a user, I want processing failures to preserve completed stages, so that I
    do not need to start over.
19. As a user, I want an inferred communication-style profile, so that I can see
    how the application interpreted my messages.
20. As a user, I want OCEAN-inspired controls presented as style dimensions, so
    that I can adjust the capsule without receiving a clinical label.
21. As a user, I want to edit every inferred trait and descriptor, so that I
    remain the authority over my identity.
22. As a user, I want to see representative phrases or evidence for style
    descriptors, so that the result is explainable.
23. As a user, I want topic content separated from writing style where possible,
    so that the capsule does not simply repeat what I talked about.
24. As a user, I want to approve the small set of private style exemplars retained
    by my capsule, so that live steering can be derived from data I control.
25. As a user, I want MiniCPM to derive my steering direction during the live
    generation workflow, so that the effect comes from my approved data rather
    than a precomputed generic personality tensor.
26. As a user, I want MiniCPM to generate an unsteered baseline and a steered
    response to the same prompt, so that I can judge the steering effect.
27. As a user, I want to adjust steering strength, so that I can trade subtlety
    for stronger characterization.
28. As a user, I want the application to warn when extreme steering harms output
    quality, so that experimentation remains understandable.
29. As a user, I want my capsule to remember its exact MiniCPM model revision and
    vector configuration, so that compatibility is verifiable.
30. As a user, I want to save my capsule after successful extraction, so that I
    can return using only the exemplars I explicitly approved rather than the
    complete raw chat export.
31. As a user, I want card art generated from my approved profile, so that the
    visual represents the personality I accepted.
32. As a user, I want to regenerate card art with controlled variation, so that
    I can choose a card I like.
33. As a user, I want generated art to use a consistent Persona Capsule visual
    language, so that cards feel collectible.
34. As a user, I want to upload voice samples, so that my capsule can speak with
    an Instant Voice Clone.
35. As a user, I want clear recording guidance, so that the clone has usable
    source audio.
36. As a user, I want to explicitly confirm that the uploaded voice is mine or
    that I have permission to clone it, so that cloning is consent-based.
37. As a user, I want the application to report whether ElevenLabs requires
    verification, so that I understand why cloning may not complete.
38. As a user, I want to generate a short signature voice line, so that the card
    has an immediately playable audio sample.
39. As a user, I want the default voice clone to be temporary, so that it is not
    retained unnecessarily.
40. As a user, I want to opt into retaining a live voice clone, so that future
    capsule responses can be synthesized in that voice.
41. As a user, I want retained voice clones deleted when I delete my capsule, so
    that account-level deletion is complete.
42. As a user, I want to chat with my capsule, so that I can experience the
    combined profile, steering, and generation behavior.
43. As a user, I want generated text read aloud, so that the text and voice
    modalities feel connected.
44. As a user, I want to choose another compatible capsule for fusion, so that I
    can create a hybrid personality.
45. As a user, I want a percentage control for fusion, so that I can explore
    different blends.
46. As a user, I want fusion to normalize vector magnitudes, so that a displayed
    60/40 blend is not silently dominated by one large vector.
47. As a user, I want incompatible capsules rejected with an explanation, so
    that invalid model vectors are never combined.
48. As a user, I want fused text descriptors and card art, so that the result is
    a complete collectible rather than only hidden tensor math.
49. As a user, I want to select which source voice speaks for a fusion, so that
    the application does not pretend that ElevenLabs voices support vector
    interpolation.
50. As a user, I want to save a fusion as a new capsule, so that combinations
    become reusable objects.
51. As a user, I want to challenge another capsule to a battle, so that the
    social experience has an entertaining competitive loop.
52. As a user, I want both capsules to answer the same hidden challenge, so that
    the comparison is fair.
53. As a user, I want Nemotron to judge anonymized responses, so that names and
    popularity do not bias the result.
54. As a user, I want the judge order reversed and aggregated, so that positional
    bias is reduced.
55. As a user, I want battle scores separated into style fidelity, response
    quality, and safety, so that one unexplained number is not treated as truth.
56. As a user, I want to read a short judge rationale, so that battle outcomes
    feel meaningful.
57. As a user, I want battle results clearly labelled as game evaluation, so that
    they are not confused with psychological measurement.
58. As a user, I want my capsule private by default, so that creation never
    automatically exposes personal information.
59. As a user, I want to preview exactly what will become public, so that I can
    remove fields before publishing.
60. As a user, I want separate visibility controls for profile text, examples,
    card image, and voice sample, so that sharing is granular.
61. As a user, I want to publish a capsule with a stable URL, so that I can share
    it outside the application.
62. As a user, I want a published card page to work without login, so that X
    crawlers and friends can see it.
63. As a user, I want the shared page to contain a 1200x628 preview image and
    accurate Open Graph metadata, so that the X post looks intentional.
64. As a user, I want an X share action with prepared text, so that sharing takes
    one step.
65. As a user, I want viewers to interact with a safe public demo of my capsule,
    so that the share can lead to engagement.
66. As a user, I want to unpublish a capsule, so that its public page stops
    exposing content.
67. As a user, I want to delete a capsule, so that its data and artifacts are
    removed from the system.
68. As a user, I want deletion status for external resources, so that failures
    such as an ElevenLabs deletion error are visible and retried.
69. As a user, I want to export a `.persona` file, so that the human-readable
    profile can be used outside this application.
70. As a technical user, I want a manifest describing the exact model and vector
    compatibility, so that I can understand what the capsule can actually run on.
71. As a technical user, I want optional LoRA model artifacts exported in safe tensor formats rather than unsafe arbitrary pickles, so that sharing does not require executing serialized Python objects.
72. As a technical user, I want integrity hashes for exported artifacts, so that
    corruption or substitution can be detected.
73. As a Deep Capsule user, I want personal MiniCPM LoRA training to run
    asynchronously on Modal, so that the Space does not remain blocked.
74. As a Deep Capsule user, I want training progress and checkpoint status, so
    that I know whether the job is advancing.
75. As a Deep Capsule user, I want the application to compare the LoRA against
    baseline and steering-only generation, so that training has measurable value.
76. As a Deep Capsule user, I want the best checkpoint selected using held-out
    examples rather than training loss alone, so that memorization is discouraged.
77. As a Deep Capsule user, I want optional personal visual-LoRA training, so
    that future card art remains visually consistent.
78. As a Deep Capsule user, I want visual training to be optional, so that I do
    not spend time and compute when the global card style is sufficient.
79. As a Deep Capsule user, I want failed training jobs to leave the Quick
    Capsule intact, so that enrichment never destroys a working capsule.
80. As an administrator, I want external API keys stored only in platform secret
    stores, so that credentials never enter source control or client responses.
81. As an administrator, I want per-user and global quotas, so that one user
    cannot exhaust ZeroGPU, Modal, FLUX, or ElevenLabs resources.
82. As an administrator, I want cost estimates before Deep Capsule training, so
    that expensive jobs are deliberate.
83. As an administrator, I want to disable optional services independently, so
    that the core demo survives provider outages.
84. As an administrator, I want auditable lifecycle events without raw personal
    content, so that failures can be diagnosed without retaining user messages.
85. As a judge, I want a pre-built demo path, so that I can evaluate the product
    even if I do not upload personal data.
86. As a judge, I want the README to identify where MiniCPM, Nemotron, Modal,
    FLUX, ElevenLabs, and Codex are used, so that sponsor contributions are clear.
87. As a judge, I want a demo video covering the complete path, so that temporary
    GPU or API exhaustion does not prevent evaluation.
88. As a judge, I want limitations and measured results documented, so that the
    technical claims are credible.

## Implementation Decisions

### Product Positioning

- The product is described as a communication-style capsule, not a validated
  psychological assessment.
- OCEAN dimensions are used as understandable, editable style controls.
- The card is the primary product object and the manifest is its technical
  representation.
- Quick Capsule is the default judged flow; Deep Capsule is an optional
  asynchronous enrichment path.
- All users must authenticate with Hugging Face before creating, modifying,
  publishing, exporting private artifacts, or deleting capsules.
- Public published pages remain viewable without authentication.

### Application Surface

- The submitted application will be a Gradio SDK Space in the
  `build-small-hackathon` organization.
- The Space README will enable Hugging Face OAuth.
- The application will use `gr.Server` with a custom frontend and FastAPI routes
  while preserving a Gradio application interface.
- Hugging Face OAuth profile information will provide the stable creator
  identity.
- User OAuth tokens will only be requested when an action genuinely needs to
  operate on the user's Hugging Face account.
- The UI will include creation, profile review, card, chat, fusion, battle,
  library, publishing, export, and deletion surfaces.

### Capsule Creation State Machine

- Capsule creation will be represented as a resumable state machine rather than
  one long request.
- Stages will include input validation, redaction, profile inference, exemplar
  approval, art generation, optional voice cloning, assembly, evaluation, and
  persistence.
- Steering-vector extraction is part of the MiniCPM generation request, not a
  capsule-creation stage that produces a permanent user tensor.
- Each stage will record status, sanitized error information, timestamps, and
  artifact references.
- Completed stages will be reusable when a later provider call fails.
- Idempotency keys will prevent duplicate Modal jobs, voice clones, and public
  records when clients retry.

### Message Ingestion

- The initial release will support pasted messages and a narrowly defined chat
  export format.
- Import adapters will produce a normalized list of message records.
- The user must select or confirm which participant identity is theirs.
- The application will reject unsupported, empty, or clearly insufficient
  samples with actionable guidance.
- A redaction module will identify common PII patterns before model processing.
- Users will review and approve the cleaned sample.
- The full raw input and original export will be held only for the active
  processing job and deleted by default after profile and exemplar approval or
  when the job expires.
- Only the small set of messages the user explicitly approves as private style
  exemplars, together with their approved neutral contrasts, may be retained to
  support future inference-time steering.
- Logs and telemetry will never contain raw messages or voice bytes.

### Style Profile

- MiniCPM will produce a structured candidate profile from the cleaned messages.
- The profile will include style dimensions, lexical tendencies, sentence
  rhythm, humor, directness, warmth, formality, examples, and uncertainty.
- Topic and factual identity claims will be excluded from style descriptors
  unless explicitly approved by the user.
- Every inferred field will be editable before it becomes canonical capsule
  data.
- The approved profile, not the unreviewed inference, drives public card content.

### Steering Architecture

- MiniCPM4.1-8B is the canonical steering and text-generation model.
- Steering compatibility is exact-model-specific and will never be advertised
  as universal across decoder-only models.
- The system will pin the MiniCPM repository revision, Transformers version,
  tokenizer revision, steering library version, layer type, and layer indices.
- A capsule will store approved positive style exemplars and semantically
  equivalent neutral contrasts, not a precomputed user activation tensor.
- Neutralization must preserve meaning while reducing vocabulary, punctuation,
  cadence, humor, and other user-specific surface choices.
- At the start of a steered generation request, the runtime will execute the
  contrastive exemplar pairs through MiniCPM, read activations from the selected
  layers, aggregate the differences, normalize each layer, and immediately
  install forward hooks for the requested generation.
- The hooks and derived activation tensors will be removed after the request.
- An optional bounded in-memory cache may reuse a derived vector only within a
  warm process when its capsule version, exemplar hash, model revision, layer
  recipe, and steering configuration match exactly.
- In-memory cache entries will expire, will never be written to persistent
  storage, and will be invalidated whenever the user edits an exemplar.
- A development-only global OCEAN-inspired contrast set may be used to evaluate
  and explain individual style dimensions, but it will not replace the user's
  own exemplar-derived steering in the product's main generation path.
- Vector tensors will be normalized per layer before fusion or strength scaling.
- The steering engine will expose a small stable interface: validate exemplars,
  derive a request-scoped vector, compose compatible request-scoped vectors,
  apply hooks for generation, and guarantee cleanup.
- Capsule JSON will store the model recipe, exemplar hashes, calibration values,
  and format version required to reproduce derivation.
- User steering tensors will not be exported or persisted.
- Unsupported library conveniences such as direct arithmetic on
  `SteeringVector` objects or `.save()` will be implemented in the product's own
  request-scoped adapter layer rather than assumed.

### MiniCPM Generation And LoRA

- MiniCPM inference will run on ZeroGPU for the primary submitted experience.
- Generation will support baseline, live exemplar-derived steering, and optional
  LoRA-plus-live-steering modes.
- Quick Capsule will not require LoRA training.
- Deep Capsule may train a small personal writing LoRA on Modal.
- The initial LoRA experiment will use conservative rank, target modules, and
  training duration, but parameters will be selected from measured spikes rather
  than treated as guaranteed.
- Training will split examples into training and held-out evaluation sets.
- The system will compare base, live-steering-only, LoRA-only, and combined
  behavior.
- The LoRA will only be attached to a capsule when it improves held-out style
  fidelity without unacceptable memorization or quality loss.
- Published Well-Tuned claims will require that the submitted app actually uses
  the published fine-tuned artifact.

### FLUX Visual System

- FLUX.2 Klein 4B will generate card art and share images.
- A global Persona Capsule visual-style LoRA will be trained on Modal using the
  FLUX.2 Klein base model.
- Distilled FLUX inference will be used only after compatibility with the
  selected LoRA checkpoint is verified.
- The global LoRA will establish card composition, texture, framing, and visual
  consistency.
- Approved personality descriptors will map to prompt elements such as palette,
  setting, symbols, pose, lighting, and energy.
- The user can regenerate controlled variations without changing the canonical
  style profile.
- Deep Capsule may optionally generate a curated synthetic image set and train a
  capsule-specific visual LoRA for consistent future art.
- Personal visual training will run asynchronously and will not block Quick
  Capsule creation.
- Public card images will be rendered separately in interactive card and
  1200x628 social-preview formats.

### Voice Cloning

- Voice cloning will use ElevenLabs Instant Voice Cloning.
- The backend will create a clone through `POST /v1/voices/add` or the equivalent
  official SDK method and store the returned `voice_id`.
- Text-to-speech will use the ordinary ElevenLabs conversion API with that
  `voice_id`.
- Voice cloning requires an explicit ownership and consent confirmation.
- Audio quality guidance will be shown before upload.
- Verification-required responses will be surfaced honestly and will not be
  treated as successful clones.
- The default path will generate a signature sample and delete the temporary
  clone after the session or configured retention window.
- A user may explicitly opt into retaining a clone for live capsule speech.
- Retained clones will be referenced by provider ID, never exported as model
  weights.
- Capsule deletion will issue an ElevenLabs voice deletion request and track
  retries if the provider is unavailable.
- Fusion will not claim to interpolate cloned voice embeddings. Users will
  select a source voice or use alternating source voices.

### Nemotron Battle And Evaluation

- NVIDIA Nemotron 3 Nano 4B will perform battle judging and selected offline
  evaluations.
- Nemotron inference will run on Modal to isolate its dependency and memory
  profile from the ZeroGPU Space.
- The judge will receive anonymized candidate labels and the same challenge,
  rubric, and approved style evidence for each capsule.
- The system will run both candidate orders and aggregate the results.
- Scores will be separated into style fidelity, response quality, instruction
  adherence, and safety.
- The judge prompt will treat candidate content as untrusted data and resist
  prompt injection.
- Battle output will be framed as an entertaining model judgment, not an
  objective personality measurement.
- Evaluation records will store prompts, sanitized outputs, model revision,
  rubric version, and score details needed for reproducibility.

### Fusion

- Fusion is supported only between capsules that share the exact model revision,
  hidden size, layer type, layer set, aggregation method, and normalization
  recipe.
- During a fused generation request, the runtime will derive each source
  capsule's vector from its approved exemplars, normalize both vectors per layer,
  and combine them using the selected weights before installing generation
  hooks.
- No fused steering tensor will be persisted; the fusion stores its source
  references, source versions, weights, and derivation recipe.
- Profile descriptors will be merged through a deterministic schema-aware
  operation followed by a model-generated summary that the user can edit.
- A fused capsule will receive newly generated card art.
- Voice will be explicitly selected from an existing permitted source rather
  than mathematically blended.
- A fusion can be saved as a new capsule with both source capsule IDs and
  composition weights in its provenance.
- Deleting a source capsule will not silently destroy an already-materialized
  fusion, but provenance will indicate that the source is no longer available.

### Capsule Data Model

- A capsule record will contain an immutable ID, creator Hugging Face ID,
  display name, creation mode, lifecycle state, timestamps, and visibility.
- It will contain the approved style profile and user-controlled public fields.
- It will contain the approved private style exemplars and neutral contrasts
  required for runtime steering.
- It will reference the steering recipe, LoRA artifacts, image assets, voice
  assets or provider IDs, evaluation results, and exports.
- It will contain exact model IDs, revisions, licenses, library versions, vector
  layer configuration, and integrity hashes.
- It will contain consent records and retention choices without storing raw
  source content.
- It will contain provenance for fusions and Deep Capsule enrichments.
- Public and private projections will be generated from one canonical record so
  private fields cannot accidentally leak into share pages.

### Persistence

- Persistent storage will be accessed through a storage abstraction so the
  application is not coupled to ephemeral Space disk.
- The first implementation will use a private Hugging Face dataset repository
  controlled by the project as the source of truth for capsule metadata and
  ordinary generated artifacts.
- Large reusable or user-specific model artifacts will use dedicated private
  Hugging Face model repositories when storing them in the dataset repository
  would make normal capsule operations impractical.
- The Space's local disk will be treated as a cache, not the source of truth.
- Private objects will be keyed by opaque capsule IDs and authorized against the
  Hugging Face creator identity.
- Published projections and social images will be separately addressable for
  unauthenticated access through the application, while their underlying
  canonical records remain private.
- Complete chat exports and unselected raw messages will not enter persistent
  capsule storage.
- Approved style exemplars and neutral contrasts are private capsule data and
  will never be included in a public projection unless the user separately
  selects a specific example for publication.
- Storage operations will support create, fetch, list-by-owner, publish,
  unpublish, and delete.

### Publishing And X Sharing

- Capsules are private by default.
- Publishing requires a preview and explicit confirmation.
- The user can choose which profile fields, examples, art, and audio sample are
  public.
- Each published capsule will receive an unguessable stable slug.
- A public FastAPI route will return server-rendered HTML with capsule-specific
  Open Graph and X metadata.
- The social image will be a static 1200x628 asset accessible to crawlers.
- Public pages will never expose private exemplars, provider
  tokens, internal artifact paths, or retained voice IDs.
- The X action will prepare share text and the public capsule URL.
- Unpublishing will disable the public page and remove it from listings while
  preserving the private capsule.

### Export

- The `.persona` export will contain the human-readable approved identity,
  communication style, examples selected for export, permissions, and
  attribution metadata.
- The technical export will contain the approved steering recipe and, only when
  explicitly selected, the private exemplar/contrast pairs needed to reproduce
  live vector derivation.
- A manifest will connect the profile and optional LoRA artifacts and state exact
  MiniCPM compatibility requirements.
- The export will not include a serialized user steering tensor or claim that
  the recipe works with arbitrary models.
- Exported files will never contain API keys or internal storage credentials.

### Privacy And Safety

- Authentication is mandatory for all creator operations.
- Users must confirm rights to submitted text and audio.
- Complete raw chats, unselected messages, source exports, and source audio will
  be deleted after processing by default.
- Approved style exemplars are retained as private capsule inputs only after a
  dedicated confirmation step and can be edited or deleted by the owner.
- Public sharing is always opt-in.
- Profile inference is editable and described as uncertain.
- The product will not encourage cloning a third party without permission.
- Generated speech will be identifiable as synthetic on public pages.
- Public capsule interactions will use bounded prompts and safety controls.
- The application will support unpublish, full deletion, and external deletion
  retry states.
- Telemetry will record operational metadata but not raw source material.

### Infrastructure

- The Hugging Face Space will host the authenticated Gradio application and
  custom web routes.
- ZeroGPU will be used for MiniCPM and FLUX inference where verified practical.
- MiniCPM and FLUX may use separate GPU-decorated functions so they are not
  resident simultaneously.
- Modal will run MiniCPM LoRA training, FLUX LoRA training, Nemotron inference,
  and long-running asynchronous jobs.
- Modal Volumes will store training checkpoints and reusable model caches.
- Provider tokens will be stored using Hugging Face Space Secrets and Modal
  Secrets.
- The architecture will not depend on storing all model weights on the default
  Space disk.
- Warmup, cold-start, timeout, queue, and provider-unavailable states will have
  visible user feedback.
- Optional services will degrade independently: voice, battle, Deep training,
  and visual regeneration may fail without disabling existing capsule access.

### Cost And Quotas

- The project will operate within the supplied Modal credit allocation.
- Every long-running job will record estimated and actual duration.
- Per-user quotas will limit capsule creation, image regenerations, battles,
  voice cloning, and Deep training.
- Deep Capsule will show a compute estimate before submission.
- Administrative kill switches will disable expensive features.
- The design will prioritize one excellent demo flow over unrestricted public
  scale during the hackathon.

### Credentials And Configuration

- The repository will include documented environment-variable names and
  deployment instructions, but no real credentials.
- Local development will use a git-ignored `.env` file with permissions limited
  to the local user.
- The committed `.env.example` will contain variable names only.
- Expected variables include `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`,
  `ELEVENLABS_API_KEY`, and any optional augmentation-provider key.
- Deployment credentials will be copied into Hugging Face Space Secrets and
  named Modal Secrets rather than shipped in the application image.
- Startup validation will report missing optional and required configuration
  without printing secret values.
- Any credential pasted into chat, logs, screenshots, issues, or commits will be
  considered exposed and rotated before production or public deployment.

### Major Modules

1. **Identity Gateway**  
   Resolves Hugging Face OAuth identity and enforces creator authorization.

2. **Ingestion and Privacy Pipeline**  
   Parses messages, selects speaker content, redacts sensitive data, records
   consent, and enforces raw-data deletion.

3. **Style Profiler**  
   Produces structured, editable communication-style profiles and uncertainty.

4. **Steering Engine**  
   Owns live contrastive extraction, normalization, compatibility,
   request-scoped composition, hook application, cleanup, and ephemeral caching.

5. **MiniCPM Runtime**  
   Provides baseline and steered generation through ZeroGPU.

6. **Training Orchestrator**  
   Starts, observes, retries, and cancels Modal LoRA jobs without coupling the
   UI to training implementation.

7. **Visual Capsule Engine**  
   Maps profiles to FLUX prompts, invokes global or personal LoRAs, and renders
   interactive and social card assets.

8. **Voice Gateway**  
   Creates ElevenLabs clones, synthesizes speech, applies retention choices, and
   deletes provider resources.

9. **Battle Judge**  
   Builds blinded rubrics, invokes Nemotron on Modal, swaps candidate order, and
   aggregates structured results.

10. **Fusion Engine**  
    Validates compatibility and combines profile, runtime-derived steering,
    visual, voice selection, and provenance.

11. **Capsule Repository**  
    Stores canonical private records and exposes authorized lifecycle operations.

12. **Publishing Gateway**  
    Produces public projections, share pages, social metadata, and unpublishing.

13. **Export Engine**  
    Generates `.persona`, model-artifact, and compatibility-manifest exports.

14. **Workflow Coordinator**  
    Implements resumable Quick and Deep Capsule state machines.

15. **Operations Layer**  
    Provides quotas, cost records, structured telemetry, health checks, and
    provider feature flags.

## Testing Decisions

### Testing Principles

- Tests will assert externally observable behavior rather than private
  implementation details.
- Model and provider calls will be represented by contract-compatible fakes in
  most automated tests.
- A small number of opt-in integration tests will call real providers when
  credentials and budget are available.
- Golden data will contain synthetic messages and synthetic audio, never a
  contributor's private exports.
- Probabilistic model quality will be tested with bounded acceptance criteria,
  structured schemas, and regression examples rather than exact text matching.
- Every destructive lifecycle operation will have idempotency and retry tests.

### Unit And Contract Coverage

1. **Identity Gateway**
   - Reject unauthenticated creator operations.
   - Permit only the owner to read or mutate private capsules.
   - Permit unauthenticated access only to published projections.

2. **Ingestion and Privacy Pipeline**
   - Parse supported inputs.
   - Select only the confirmed user's messages.
   - Redact configured PII patterns.
   - Reject insufficient or malformed samples.
   - Delete raw data after success, failure timeout, and explicit cancellation.

3. **Style Profiler**
   - Validate structured profile output.
   - Preserve user edits as canonical values.
   - Keep private evidence out of public projections.

4. **Steering Engine**
   - Detect model, revision, hidden-size, layer, aggregation, and recipe
     incompatibility.
   - Derive vectors from approved exemplar/contrast pairs during a request.
   - Normalize per-layer vectors correctly.
   - Compose request-scoped weights deterministically.
   - Reuse only exact-match, unexpired in-memory cache entries.
   - Invalidate cached vectors after an exemplar edit.
   - Never write user steering tensors to persistent storage.
   - Apply zero-strength steering as a baseline-equivalent operation.

5. **MiniCPM Runtime**
   - Produce baseline and live-steering modes.
   - Respect generation limits.
   - Release steering hooks after each request.
   - Release request-scoped activation tensors after each request or cache expiry.
   - Prevent one user's vector from leaking into another request.

6. **Training Orchestrator**
   - Create one Modal job per idempotency key.
   - Resume status polling.
   - Preserve Quick Capsule after training failure.
   - Attach only validated checkpoints.

7. **Visual Capsule Engine**
   - Generate deterministic prompt structures from approved profiles.
   - Keep private text out of image prompts.
   - Produce required image dimensions.
   - Fall back to the global style when personal training is unavailable.

8. **Voice Gateway**
   - Require consent before clone creation.
   - Handle clone success, verification-required, quota, and provider errors.
   - Use the returned voice ID for synthesis.
   - Delete temporary and capsule-owned retained clones idempotently.

9. **Battle Judge**
   - Send identical challenges to both candidates.
   - Remove identifying labels.
   - Run both candidate orders.
   - Validate structured judge output.
   - Neutralize prompt injection inside candidate responses.

10. **Fusion Engine**
    - Reject incompatible capsules.
    - Derive, normalize, and combine compatible vectors within the fused
      generation request.
    - Preserve exact source weights and provenance.
    - Persist no fused user steering tensor.
    - Require an explicit voice-selection strategy.

11. **Capsule Repository**
    - Enforce owner-scoped listing and retrieval.
    - Separate canonical private and public projections.
    - Make create, publish, unpublish, and delete operations idempotent.

12. **Publishing Gateway**
    - Render capsule-specific Open Graph metadata.
    - Serve the correct 1200x628 image.
    - Exclude private fields and internal references.
    - Return unavailable after unpublishing or deletion.

13. **Export Engine**
    - Validate `.persona` JSON.
    - Include exact model compatibility metadata.
    - Exclude secrets and private fields not selected for export.
    - Export no user activation tensors.
    - Verify exemplar and manifest integrity hashes.

14. **Workflow Coordinator**
    - Resume from each stage boundary.
    - Avoid duplicate external resources after retries.
    - Expose understandable partial-success states.

15. **Operations Layer**
    - Enforce user quotas.
    - Disable individual providers through feature flags.
    - Record cost and latency without recording raw content.

### Integration Tests

- Hugging Face OAuth profile retrieval in a staging Space.
- MiniCPM load and steering-layer discovery on the pinned runtime.
- Live exemplar-derived steering extraction and generation on ZeroGPU.
- Hook and activation cleanup after successful and failed generations.
- Warm-session cache reuse and invalidation without persistent tensor writes.
- FLUX global LoRA inference on the chosen checkpoint.
- Modal Nemotron structured battle response.
- Modal MiniCPM and FLUX training job smoke tests.
- ElevenLabs test clone creation, speech generation, and clone deletion using
  synthetic authorized audio.
- Public share-page crawl and metadata validation.
- End-to-end capsule deletion across storage and external providers.

### Quality Evaluation

- Build a synthetic benchmark of writing samples with held-out prompts.
- Compare base, live-steering-only, LoRA-only, and combined configurations.
- Measure style-rubric adherence, semantic quality, repetition, and memorization.
- Test multiple steering strengths and choose bounded defaults.
- Test fusion at several weights and inspect whether both source characteristics
  remain visible.
- Record evaluation limitations in the README and demo narrative.

### Manual Acceptance Testing

- Desktop and mobile card creation.
- OAuth login and logout.
- Quick Capsule from a clean sample.
- Quick Capsule with recoverable provider failures.
- Voice consent, creation, playback, retention, and deletion.
- Fusion creation and incompatible-fusion error.
- Battle flow and reversed-order result.
- Publish, X preview, unpublish, and delete.
- Cold-start behavior and visible queue states.
- Full demo flow using pre-built capsules when live quotas are exhausted.

## Out of Scope

- Anonymous capsule creation or anonymous private storage.
- Claims of clinical, diagnostic, or scientifically validated personality
  assessment.
- Universal steering vectors that work across arbitrary language models.
- Mathematical interpolation of ElevenLabs voice clones.
- Automatic publishing of user data without explicit confirmation.
- Unrestricted public-scale operation beyond hackathon quotas.
- Full support for every chat-export format.
- Real-time collaborative capsule editing.
- Native mobile applications.
- Cryptocurrency, NFTs, payments, or marketplace functionality.
- Full moderation tooling for a large public social network.
- Guaranteed per-user LoRA completion during the live Quick Capsule flow.
- Guaranteed multilingual steering without measured language-specific evidence.
- Tiny Titan and Off the Grid eligibility while MiniCPM4.1-8B, Modal, ZeroGPU,
  and ElevenLabs remain core dependencies.
- A claim that the PersonaSpec draft is broadly supported by third-party
  platforms.
- Automatic cloning of public figures or people other than the consenting user.

## Further Notes

### Recommended Build Priority

The implementation must be ordered by technical risk:

1. Prove MiniCPM plus steering on one trait and selected layers.
2. Implement request-scoped steering derivation, cleanup, and compatible fusion.
3. Build the authenticated Quick Capsule workflow.
4. Generate a polished card using the global FLUX LoRA.
5. Add publishing and X-compatible share pages.
6. Add ElevenLabs voice creation and deletion.
7. Add Nemotron battle through Modal.
8. Add Deep Capsule training only after the complete Quick path works.

### Delivery Schedule

#### June 11, 2026

- Finalize PRD and repository structure.
- Pin core versions.
- Run MiniCPM steering compatibility spike.
- Define capsule schema and storage interface.

#### June 12, 2026

- Implement OAuth, ingestion, profile review, steering, and Quick Capsule state.
- Implement baseline versus steered generation.
- Begin global FLUX LoRA training.

#### June 13, 2026

- Complete card UI, FLUX inference, capsule library, export, and fusion.
- Add private persistence and deletion.
- Implement public share routes and social images.

#### June 14, 2026

- Add ElevenLabs voice lifecycle.
- Add Nemotron battle on Modal.
- Deploy and stabilize the hackathon Space.
- Run full acceptance tests and record fallback demo assets.

#### June 15, 2026

- Freeze feature development.
- Record and publish the demo video.
- Publish the social post.
- Complete README, sponsor explanation, frontmatter tags, and submission checks.
- Fix only submission-blocking defects.

### Submission Positioning

- Enter Thousand Token Wood as the primary track.
- Enter the OpenBMB sponsor category because MiniCPM is the core runtime and
  steering target.
- Enter the NVIDIA category because Nemotron performs a visible core battle and
  evaluation role.
- Enter the Modal category because training and Nemotron runtime use Modal.
- Enter the OpenAI Codex category through Codex-attributed repository commits and
  documented use throughout architecture, implementation, testing, and
  deployment.
- Pursue Off Brand through the custom `gr.Server` card experience.
- Pursue Best Demo through a polished end-to-end narrative.
- Pursue Well-Tuned only if the submitted app uses a published fine-tuned model.
- Pursue Sharing is Caring and Field Notes through a public development trace
  and technical write-up.
- Do not claim Tiny Titan, Off the Grid, or Llama Champion unless the actual
  shipped application later satisfies those requirements.

### Required Secrets At Deployment

- Hugging Face service token with the minimum repository permissions needed by
  the selected storage and publishing design.
- Modal authentication and named Modal secrets.
- ElevenLabs API key with Instant Voice Cloning access.
- Optional augmentation-provider key if synthetic training-pair generation uses
  an external API.

Rotated deployment secrets will be installed only after repository and
deployment configuration are ready and will never be committed. The initial
values supplied during planning must be rotated before deployment because they
were transmitted in chat.

### Review Gate

This document must be approved before:

- creating the public GitHub repository;
- submitting the PRD as a GitHub issue;
- creating the Hugging Face Space;
- provisioning persistent repositories or storage;
- adding rotated credentials to Hugging Face and Modal secret stores;
- beginning implementation.
