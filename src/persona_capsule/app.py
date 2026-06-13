"""FastAPI boundary and mounted Persona Capsule Gradio application."""

from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from persona_capsule.config import Settings
from persona_capsule.identity import IdentityGateway
from persona_capsule.library import CapsuleLibrary
from persona_capsule.modal_gateway import ModalSteeringGateway
from persona_capsule.repository import CapsuleRepository, InMemoryCapsuleRepository
from persona_capsule.repository_factory import build_capsule_repository
from persona_capsule.steering_service import CapsuleSteeringService, SteeringGateway
from persona_capsule.ui import CSS, build_demo, build_theme


def create_app(
    settings: Settings | None = None,
    repository: CapsuleRepository | None = None,
    steering_gateway: SteeringGateway | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if repository is None:
        repository = (
            InMemoryCapsuleRepository()
            if settings.app_env == "test"
            else build_capsule_repository(settings)
        )
    identity_gateway = IdentityGateway(settings)
    capsule_library = CapsuleLibrary(repository)
    steering_service = CapsuleSteeringService(
        capsule_library,
        steering_gateway or ModalSteeringGateway(),
    )
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

    @app.get("/api/creator/capsules", tags=["creator"])
    async def list_creator_capsules() -> dict[str, Any]:
        principal = identity_gateway.resolve_local()
        if principal is None:
            raise HTTPException(status_code=401, detail="Hugging Face login required")
        records = capsule_library.list_capsules(principal)
        return {
            "owner": principal.username,
            "capsules": [
                {
                    "capsule_id": record.capsule_id,
                    "name": record.name,
                    "status": record.status,
                }
                for record in records
            ],
        }

    return gr.mount_gradio_app(
        app=app,
        blocks=build_demo(
            settings,
            identity_gateway,
            capsule_library,
            steering_service,
        ),
        path="/app",
        footer_links=[],
        theme=build_theme(),
        css=CSS,
    )


app = create_app()
