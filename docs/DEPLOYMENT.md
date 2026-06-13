# Deployment

## Safety Gate

Credentials previously pasted into chat or logs are compromised. Rotate the
Hugging Face, Modal, and ElevenLabs credentials before any public deployment.
Do not reuse the original values.

Create a separate Hugging Face service token with only the access required for
the private capsule Dataset and private Deep Capsule model repositories. Put the
rotated values in the ignored local `.env`, then set:

```dotenv
CONFIRM_CREDENTIALS_ROTATED=true
SPACE_HF_SERVICE_TOKEN=...
```

## Modal

Review the dry configuration, then deploy all independent GPU services:

```bash
./scripts/deploy_modal.sh
```

The script installs the Hugging Face service token in the named Modal Secret
`persona-capsule-huggingface`. It never commits credentials.

## Hugging Face Space

First inspect the deployment plan:

```bash
uv run python scripts/deploy_space.py --dry-run
```

Then create/update
`build-small-hackathon/persona-capsule`, install Space Secrets and Variables,
upload the repository with private paths excluded, and request ZeroGPU:

```bash
uv run python scripts/deploy_space.py
```

The Space is public, while capsule persistence and Deep Capsule adapters remain
private. Hugging Face OAuth is enabled by README frontmatter.

## Release Gate

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv lock --check
uv run python scripts/check_submission.py
```

After owner review adds the public demo-video and social-post links to the
README, run the final gate:

```bash
uv run python scripts/check_submission.py --strict
```
