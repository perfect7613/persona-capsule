# Persona Capsule

Turn your communication style into a portable, composable, collectible persona
card.

Persona Capsule is being built for the Hugging Face Build Small Hackathon 2026.
It combines:

- inference-time MiniCPM4.1-8B activation steering;
- FLUX.2 Klein personality card art;
- ElevenLabs Instant Voice Cloning;
- capsule fusion and Nemotron-judged battles;
- authenticated private capsules and X-compatible public sharing;
- ZeroGPU hosting with Modal training and inference jobs.

## Status

Planning is complete and implementation has not started.

- [Product requirements](./PRD.md)
- [Technical plan audit](./PLAN_AUDIT.md)

The final Gradio application will be deployed to a Space in the
[`build-small-hackathon`](https://huggingface.co/build-small-hackathon)
organization.

## Core Technical Decision

User steering vectors are derived from approved exemplar and neutral-contrast
pairs during MiniCPM inference. User activation tensors are request-scoped and
are not written to persistent storage.

## Local Configuration

Copy `.env.example` to `.env` and supply rotated development credentials:

```bash
cp .env.example .env
```

Never commit `.env`. Production credentials will use Hugging Face Space Secrets
and Modal Secrets.

## Security

Do not submit credentials through issues, commits, logs, screenshots, or chat.
Any exposed credential must be rotated before deployment.

## License

A project license will be selected before the first implementation release.
