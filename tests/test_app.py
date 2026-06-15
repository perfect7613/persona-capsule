import re

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
            voxcpm_available=True,
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
            "voxcpm2": True,
        },
        "features": {
            "creation": True,
            "steering": False,
            "card": True,
            "voice": True,
            "fusion": False,
            "battle": False,
            "deep_training": False,
        },
        "daily_quotas": {
            "creation": 10,
            "steering": 20,
            "card": 10,
            "voice": 5,
            "fusion": 8,
            "battle": 6,
            "deep_training": 2,
            "public_chat": 40,
        },
    }


def test_root_redirects_to_demo() -> None:
    with TestClient(create_app(Settings(), steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/app/"


def test_landing_path_without_slash_uses_relative_redirect() -> None:
    with TestClient(create_app(Settings(), steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/app", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/app/"


def test_oauth_root_routes_bridge_to_mounted_gradio_app() -> None:
    app = create_app(
        Settings(space_environment=True),
        steering_gateway=UnusedSteeringGateway(),
    )

    with TestClient(app) as client:
        login = client.get(
            "/login/huggingface?_target_url=%2Fapp%2F%3Fstep%3Dcreate",
            follow_redirects=False,
        )
        callback = client.get(
            "/login/callback?code=oauth-code&state=oauth-state",
            follow_redirects=False,
        )
        logout = client.get(
            "/logout?_target_url=%2Fapp%2F",
            follow_redirects=False,
        )

    assert login.status_code in {302, 307}
    assert login.headers["location"] == (
        "/app/login/huggingface?_target_url=%2Fapp%2F%3Fstep%3Dcreate"
    )
    assert callback.status_code in {302, 307}
    assert callback.headers["location"] == ("/app/login/callback?code=oauth-code&state=oauth-state")
    assert logout.status_code in {302, 307}
    assert logout.headers["location"] == "/app/logout?_target_url=%2Fapp%2F"


def test_landing_path_is_bootable() -> None:
    with TestClient(create_app(Settings(), steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/app/")

    assert response.status_code == 200
    assert "Persona Capsule" in response.text


def test_landing_frontend_assets_are_served_from_mount_path() -> None:
    with TestClient(create_app(Settings(), steering_gateway=UnusedSteeringGateway())) as client:
        response = client.get("/app/")
        asset_paths = re.findall(
            r'(?:src|href)="\./(assets/[^"]+)"',
            response.text,
        )

        assert asset_paths
        assert "/_app/immutable/" not in response.text
        for asset_path in asset_paths:
            asset_response = client.get(f"/app/{asset_path}")
            assert asset_response.status_code == 200
            assert asset_response.content


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
