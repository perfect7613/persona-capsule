"""FastAPI boundary and mounted Persona Capsule Gradio application."""

from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from persona_capsule.battle import BattleJudgeGateway, CapsuleBattleService
from persona_capsule.card import ArtProvider, CapsuleCardService
from persona_capsule.config import Settings
from persona_capsule.deep_gateway import ModalDeepTrainingGateway
from persona_capsule.deep_training import DeepCapsuleService, DeepTrainingGateway
from persona_capsule.flux_gateway import ModalFluxArtGateway
from persona_capsule.fusion import CapsuleFusionService, FusionGateway
from persona_capsule.identity import IdentityGateway
from persona_capsule.library import CapsuleLibrary
from persona_capsule.modal_gateway import ModalSteeringGateway
from persona_capsule.nemotron_gateway import ModalNemotronGateway
from persona_capsule.operations import (
    DailyQuotaManager,
    FeatureFlags,
    OperationsGuard,
    SafeTelemetry,
)
from persona_capsule.publishing import PublishingService
from persona_capsule.repository import CapsuleRepository, InMemoryCapsuleRepository
from persona_capsule.repository_factory import build_capsule_repository
from persona_capsule.steering_service import CapsuleSteeringService, SteeringGateway
from persona_capsule.ui import CSS, build_demo, build_theme
from persona_capsule.voice import (
    CapsuleVoiceService,
    ElevenLabsVoiceProvider,
    VoiceProvider,
)


def create_app(
    settings: Settings | None = None,
    repository: CapsuleRepository | None = None,
    steering_gateway: SteeringGateway | None = None,
    art_provider: ArtProvider | None = None,
    voice_provider: VoiceProvider | None = None,
    fusion_gateway: FusionGateway | None = None,
    battle_judge_gateway: BattleJudgeGateway | None = None,
    deep_training_gateway: DeepTrainingGateway | None = None,
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
    modal_steering_gateway = steering_gateway or ModalSteeringGateway()
    steering_service = CapsuleSteeringService(
        capsule_library,
        modal_steering_gateway,
    )
    card_service = CapsuleCardService(
        capsule_library,
        settings.capsule_data_dir,
        primary_provider=(
            art_provider
            if art_provider is not None
            else ModalFluxArtGateway()
            if settings.modal_available
            else None
        ),
    )
    publishing_service = PublishingService(
        capsule_library,
        repository,
        settings.capsule_data_dir,
        settings.public_base_url,
    )
    if voice_provider is None and settings.elevenlabs_available:
        try:
            voice_provider = ElevenLabsVoiceProvider()
        except RuntimeError:
            voice_provider = None
    voice_service = CapsuleVoiceService(
        capsule_library,
        settings.capsule_data_dir,
        voice_provider,
        temporary_hours=settings.voice_temporary_hours,
    )
    fusion_service = CapsuleFusionService(
        capsule_library,
        fusion_gateway or modal_steering_gateway,
        card_service,
    )
    battle_service = CapsuleBattleService(
        capsule_library,
        modal_steering_gateway,
        battle_judge_gateway or ModalNemotronGateway(),
    )
    deep_service = DeepCapsuleService(
        capsule_library,
        deep_training_gateway or ModalDeepTrainingGateway(),
    )
    effective_features = {
        **settings.feature_flags,
        "steering": settings.enable_steering and settings.modal_available,
        "voice": settings.enable_voice and settings.elevenlabs_available,
        "fusion": settings.enable_fusion and settings.modal_available,
        "battle": settings.enable_battle and settings.modal_available,
        "deep_training": settings.enable_deep_training and settings.modal_available,
    }
    operations = OperationsGuard(
        FeatureFlags(**effective_features),
        DailyQuotaManager(settings.quotas),
        SafeTelemetry(f"{settings.capsule_data_dir}/telemetry/events.jsonl"),
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
            "features": effective_features,
            "daily_quotas": settings.quotas,
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

    @app.get(
        "/c/{slug}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def public_capsule(slug: str) -> HTMLResponse:
        try:
            record = publishing_service.get_public(slug)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capsule unavailable") from exc
        return HTMLResponse(publishing_service.render_public_html(record))

    @app.get(
        "/c/{slug}/image",
        response_class=FileResponse,
        include_in_schema=False,
    )
    async def public_capsule_image(slug: str) -> FileResponse:
        try:
            record = publishing_service.get_public(slug)
            path = publishing_service.public_image_path(record)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capsule image unavailable") from exc
        return FileResponse(path, media_type="image/png")

    @app.get(
        "/c/{slug}/audio",
        response_class=FileResponse,
        include_in_schema=False,
    )
    async def public_capsule_audio(slug: str) -> FileResponse:
        try:
            record = publishing_service.get_public(slug)
            path = publishing_service.public_audio_path(record)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capsule audio unavailable") from exc
        return FileResponse(path, media_type="audio/mpeg")

    return gr.mount_gradio_app(
        app=app,
        blocks=build_demo(
            settings,
            identity_gateway,
            capsule_library,
            steering_service,
            card_service,
            publishing_service,
            voice_service,
            fusion_service,
            battle_service,
            deep_service,
            operations,
        ),
        path="/app",
        footer_links=[],
        theme=build_theme(),
        css=CSS,
    )


app = create_app()
