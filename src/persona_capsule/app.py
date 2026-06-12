"""FastAPI boundary and mounted Persona Capsule Gradio application."""

from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from persona_capsule.config import Settings
from persona_capsule.ui import CSS, build_demo, build_theme


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Persona Capsule",
        description="Portable, composable communication personas.",
        version="0.1.0",
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    @app.get("/healthz", tags=["operations"])
    async def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "environment": settings.app_env,
            "demo_available": True,
            "providers": settings.providers,
        }

    return gr.mount_gradio_app(
        app=app,
        blocks=build_demo(settings),
        path="/app",
        footer_links=[],
        theme=build_theme(),
        css=CSS,
    )


app = create_app()
