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

Planning is complete and implementation is underway. The first vertical slice
provides a bootable FastAPI + Gradio shell, safe provider configuration status,
and an offline deterministic capsule demo.

- [Product requirements](./PRD.md)
- [Technical plan audit](./PLAN_AUDIT.md)
- [Codex development record](./CODEX_USAGE.md)
- [Canonical PRD issue](https://github.com/perfect7613/persona-capsule/issues/1)

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

## Run Locally

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are recommended:

```bash
uv sync --extra dev
uv run persona-capsule
```

Open `http://127.0.0.1:7860/app/`. The readiness endpoint is available at
`http://127.0.0.1:7860/healthz`.

The application starts without provider credentials. Missing providers are
reported as unavailable without exposing secret values.

## Verify

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Built With Codex

OpenAI Codex is being used throughout architecture, implementation, testing,
documentation, and deployment. Codex-authored commits include:

```text
Co-authored-by: Codex <noreply@openai.com>
```

Project-level attribution rules live in [`AGENTS.md`](./AGENTS.md) and
[`.codex/config.toml`](./.codex/config.toml).

## Security

Do not submit credentials through issues, commits, logs, screenshots, or chat.
Any exposed credential must be rotated before deployment.

## License

A project license will be selected before the first implementation release.
