"""Persona Capsule application entry point."""

from __future__ import annotations

from persona_capsule.app.gradio_app import build_app
from persona_capsule.config import get_settings


def main() -> None:
    settings = get_settings()
    app = build_app(settings)
    app.launch(server_name=settings.host, server_port=settings.port)


if __name__ == "__main__":
    main()
