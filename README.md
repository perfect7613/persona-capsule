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

Planning is complete. The repository now includes:

- a Persona Capsule Gradio application (`app.py`)
- restored ai-toolkit training infrastructure (`run_modal.py`, `toolkit/`, `jobs/`, etc.)
- visual LoRA training orchestration via Modal (`persona_capsule/services/training_orchestrator.py`)

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

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements-app.txt
pip install -r requirements-dev.txt
copy .env.example .env
python app.py
```

Open `http://127.0.0.1:7860` and use the **Visual LoRA Training** tab.

## Visual LoRA training (Modal + ai-toolkit)

1. Prepare a Modal dataset volume (`my-dataset`) with images and `.txt` captions.
2. Set `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `MODAL_DATASET_PATH` in `.env`.
3. Install Modal: `pip install modal`
4. From the Gradio app, preview or submit a capsule-specific config.
5. Or run manually:

```bash
modal run run_modal.py --config /root/ai-toolkit/config/examples/persona_capsule.yaml
```

Generated capsule configs are written to `config/generated/` and job metadata to `artifacts/training_jobs/`.

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
