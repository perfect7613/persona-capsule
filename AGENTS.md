# Persona Capsule Repository Instructions

## Codex Attribution

- Every commit whose changes were authored or materially implemented by Codex
  must end with this exact Git trailer:

  `Co-authored-by: Codex <noreply@openai.com>`

- Do not add the Codex trailer to a commit containing only human-authored work.
- Preserve the human developer as the primary Git author.
- Before pushing a Codex-authored commit, verify the trailer with:

  `git log -1 --format=full`

## Credential Safety

- Never commit `.env` or real API credentials.
- Use `.env.example` for variable names only.
- Treat credentials exposed in chat, logs, issues, or commits as compromised.

## Product Source Of Truth

- Use `PRD.md` as the approved product and architecture specification.
- Use GitHub issue `#1` as the canonical public PRD discussion.
