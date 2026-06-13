from fastapi.testclient import TestClient

from persona_capsule.app import create_app
from persona_capsule.config import Settings


class UnusedSteeringGateway:
    def compare(self, **kwargs):
        raise AssertionError(f"unexpected live steering call: {kwargs}")

    def invalidate(self, **kwargs):
        raise AssertionError(f"unexpected cache invalidation: {kwargs}")


def test_health_is_ready_and_secret_safe() -> None:
    app = create_app(
        Settings(
            app_env="test",
            hugging_face_available=True,
            modal_available=False,
            elevenlabs_available=True,
        ),
        steering_gateway=UnusedSteeringGateway(),
    )

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "environment": "test",
        "demo_available": True,
        "providers": {
            "hugging_face": True,
            "modal": False,
            "elevenlabs": True,
        },
    }


def test_root_redirects_to_demo() -> None:
    with TestClient(create_app(Settings(), steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/app/"


def test_landing_path_is_bootable() -> None:
    with TestClient(create_app(Settings(), steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/app/")

    assert response.status_code == 200
    assert "Persona Capsule" in response.text


def test_creator_route_rejects_unauthenticated_requests() -> None:
    with TestClient(
        create_app(
            Settings(app_env="production"),
            steering_gateway=UnusedSteeringGateway(),
        )
    ) as client:
        response = client.get("/api/creator/capsules")

    assert response.status_code == 401
    assert response.json() == {"detail": "Hugging Face login required"}


def test_local_creator_route_is_explicit_and_owner_scoped() -> None:
    settings = Settings(
        app_env="test",
        local_identity_enabled=True,
        local_hf_username="Owner",
    )
    with TestClient(create_app(settings, steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/api/creator/capsules")

    assert response.status_code == 200
    assert response.json() == {"owner": "Owner", "capsules": []}
