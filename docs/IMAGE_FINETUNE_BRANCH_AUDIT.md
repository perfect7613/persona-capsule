# Image Fine-Tune Branch Audit

Reviewed branch: `origin/img-ftcode` at `8d223fc`.

The branch was inspected without checking it out, merging it, or copying its
vendored AI Toolkit source into the main Persona Capsule package.

## Reusable Decisions

- Fine-tuning target: `black-forest-labs/FLUX.2-klein-base-4B`.
- Product inference target: `black-forest-labs/FLUX.2-klein-4B`.
- AI Toolkit architecture key: `flux2_klein_4b`.
- Save adapter output as SafeTensors.
- Use a unique trigger token for a global Persona Capsule visual style.
- Keep training asynchronous on Modal and store outputs in a dedicated volume.
- Caption the depicted content while allowing the repeated trigger token to
  carry the learned style.

The current product runtime pins:

- inference model revision:
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`;
- fine-tuning base revision:
  `a3b4f4849157f664bdbc776fd7453c2783562f4d`.

## Not Carried Forward

- The 125,000-line vendored AI Toolkit tree is not merged into the application.
- The alternate `persona_capsule/` schema is not compatible with the approved
  `src/persona_capsule/` architecture.
- The subprocess-based Modal launcher is not used; provider calls remain behind
  typed gateways and deployed Modal classes.
- Unpinned training dependencies and generic example configurations are not
  part of the submitted Space image.
- The `Flex.2` preview configuration is unrelated to FLUX.2 Klein and is not
  used for Persona Capsule.

## Training Handoff

When Slice 11 begins, retain AI Toolkit in a dedicated training image or
external repository and send it a generated, reviewed configuration. The
application repository should own job idempotency, status, evaluation, artifact
attachment, and rollback behavior, not the training framework internals.

Official Black Forest Labs guidance recommends a diverse synthetic style
dataset, a repeated unique trigger token, and captions that describe image
content rather than restating the target style.
