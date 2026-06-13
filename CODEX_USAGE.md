# Codex Development Record

Persona Capsule is being designed and implemented with OpenAI Codex as a
substantial engineering collaborator for the Build Small Hackathon OpenAI
track.

## Codex Responsibilities

Codex has been used to:

- audit the original hackathon plan against current official documentation;
- verify MiniCPM, Nemotron, FLUX, Modal, ElevenLabs, Gradio, and ZeroGPU
  assumptions;
- identify unsupported steering-vector and portability claims;
- design the inference-time activation-steering architecture;
- define privacy, consent, deletion, authentication, and sharing behavior;
- write and review the complete product requirements document;
- establish the GitHub repository and canonical PRD issue;
- implement request-scoped MiniCPM activation extraction and hook cleanup;
- implement owner-scoped capsule storage, export, publishing, and deletion;
- integrate real FLUX and ElevenLabs provider lifecycles with safe fallbacks;
- implement weighted capsule fusion with exact recipe compatibility;
- implement blinded A/B and B/A Nemotron battles on Modal;
- implement resumable, evaluated MiniCPM QLoRA Deep Capsule jobs;
- add provider kill switches, per-user quotas, privacy-safe telemetry, tests,
  deployment tooling, and hackathon submission documentation.

## Commit Attribution

Codex-authored commits use the official trailer:

```text
Co-authored-by: Codex <noreply@openai.com>
```

The repository also contains project-scoped Codex configuration and
`AGENTS.md` guidance so attribution remains consistent through development.

## Public Evidence

- Product requirements: [PRD.md](./PRD.md)
- Technical audit: [PLAN_AUDIT.md](./PLAN_AUDIT.md)
- Canonical PRD issue:
  [perfect7613/persona-capsule#1](https://github.com/perfect7613/persona-capsule/issues/1)
- Commit history:
  [commits](https://github.com/perfect7613/persona-capsule/commits/main/)
