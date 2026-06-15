from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from persona_capsule.app import create_app
from persona_capsule.config import Settings
from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.profile import ExemplarPair, StyleDimensions, StyleProfile
from persona_capsule.publishing import PublishingService, PublishSelection
from persona_capsule.repository import (
    CapsuleNotFoundError,
    CapsuleRecord,
    InMemoryCapsuleRepository,
)

OWNER = Principal("hf:owner", "owner", "test")
PRIVATE_SENTINEL = "PRIVATE-EXEMPLAR-MUST-NOT-LEAK"


class UnusedSteeringGateway:
    def compare(self, **kwargs):
        raise AssertionError(f"unexpected steering call: {kwargs}")

    def invalidate(self, **kwargs):
        raise AssertionError(f"unexpected invalidation: {kwargs}")


class PublicChatSteeringGateway:
    def __init__(self) -> None:
        self.calls = []

    def compare(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "baseline": "A generic answer.",
            "steered": "Test the smallest reversible version first.",
            "diagnostics": {},
        }

    def invalidate(self, **kwargs):
        raise AssertionError(f"unexpected invalidation: {kwargs}")


def _record() -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id="publish-01",
        owner_id=OWNER.user_id,
        name="Clear Signal",
        status="profile_approved",
        style_profile=StyleProfile(
            summary="Warm, rigorous, and direct.",
            descriptors=("warm", "direct", "analytical"),
            lexical_tendencies=("private tendency",),
            sentence_rhythm="Short setup and clean landing.",
            dimensions=StyleDimensions(70, 80, 65, 72, 40, 90, 35),
            evidence=(PRIVATE_SENTINEL,),
            uncertainty=0.15,
        ),
        exemplar_pairs=(ExemplarPair(PRIVATE_SENTINEL, "Neutral contrast.", 0),),
        source_fingerprint="source-v1",
        social_image_ref="card/card-social.png",
        card_image_ref="card/card-interactive.png",
        voice_provider="openbmb-voxcpm2-modal",
        voice_id="private-provider-id",
        voice_status="ready",
        voice_retention="retained",
        voice_sample_ref="voice/voice-signature.wav",
        voice_model_id="eleven_multilingual_v2",
    )


def _service(tmp_path: Path):
    repository = InMemoryCapsuleRepository()
    library = CapsuleLibrary(repository)
    saved = library.save_capsule(OWNER, _record())
    owner_namespace = sha256(OWNER.user_id.encode()).hexdigest()[:24]
    image_path = tmp_path / "artifacts" / owner_namespace / saved.capsule_id / "card-social.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (1200, 628), "black").save(image_path)
    (image_path.parent / "voice-signature.wav").write_bytes(b"synthetic-audio")
    service = PublishingService(
        library,
        repository,
        tmp_path,
        "https://capsules.example",
    )
    return repository, library, service, saved


def test_publish_requires_preview_confirmation_and_keeps_stable_slug(
    tmp_path: Path,
) -> None:
    repository, _, service, record = _service(tmp_path)
    selection = PublishSelection(
        include_summary=True,
        include_descriptors=False,
        include_dimensions=False,
        include_card=True,
    )
    preview = service.preview(OWNER, record.capsule_id, selection)

    assert preview == {
        "name": "Clear Signal",
        "generated_art_label": "AI-generated artwork",
        "summary": "Warm, rigorous, and direct.",
        "social_image": True,
    }
    with pytest.raises(ValueError, match="Confirm"):
        service.publish(OWNER, record.capsule_id, selection, confirmed=False)

    first = service.publish(OWNER, record.capsule_id, selection, confirmed=True)
    second = service.publish(OWNER, record.capsule_id, selection, confirmed=True)

    assert first.public_slug == second.public_slug
    assert len(first.public_slug) >= 20
    assert repository.get_public_by_slug(first.public_slug).capsule_id == record.capsule_id
    assert PRIVATE_SENTINEL not in str(first.published_projection)

    owner_namespace = sha256(OWNER.user_id.encode()).hexdigest()[:24]
    local_image = tmp_path / "artifacts" / owner_namespace / record.capsule_id / "card-social.png"
    local_image.unlink()
    assert service.public_image_path(first).read_bytes()


def test_public_routes_render_crawler_metadata_and_unpublish(
    tmp_path: Path,
) -> None:
    repository, _, service, record = _service(tmp_path)
    published = service.publish(
        OWNER,
        record.capsule_id,
        PublishSelection(
            include_summary=True,
            include_descriptors=True,
            include_dimensions=False,
            include_card=True,
            include_voice_sample=True,
        ),
        confirmed=True,
    )
    app = create_app(
        Settings(
            app_env="test",
            capsule_data_dir=str(tmp_path),
            public_base_url="https://capsules.example",
        ),
        repository=repository,
        steering_gateway=UnusedSteeringGateway(),
    )

    with TestClient(app) as client:
        page = client.get(f"/c/{published.public_slug}")
        image = client.get(f"/c/{published.public_slug}/image")
        card = client.get(f"/c/{published.public_slug}/card")
        audio = client.get(f"/c/{published.public_slug}/audio")

    assert page.status_code == 200
    assert image.status_code == 200
    assert card.status_code == 200
    assert audio.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert card.headers["content-type"] == "image/png"
    assert audio.headers["content-type"] == "audio/wav"
    assert 'property="og:title"' in page.text
    assert 'property="og:image"' in page.text
    assert 'name="twitter:card" content="summary_large_image"' in page.text
    assert f"/c/{published.public_slug}/card" in page.text
    assert "object-fit:contain" in page.text
    assert "object-fit:cover" not in page.text
    assert f"https://capsules.example/c/{published.public_slug}" in page.text
    assert "https://x.com/intent/post" in page.text
    assert "Do%20you%20really%20know%20me%3F" in page.text
    assert "Synthetic voice" in page.text
    assert "Talk to the capsule" in page.text
    assert f"/c/{published.public_slug}/chat" in page.text
    assert "Do you really know Clear Signal?" in page.text
    assert f"/c/{published.public_slug}/challenge" in page.text
    assert f"/c/{published.public_slug}/challenge/guess" in page.text
    assert "private-provider-id" not in page.text
    assert PRIVATE_SENTINEL not in page.text
    assert OWNER.user_id not in page.text
    assert "private tendency" not in page.text

    service.unpublish(OWNER, record.capsule_id)
    with pytest.raises(CapsuleNotFoundError):
        repository.get_public_by_slug(published.public_slug)
    with TestClient(app) as client:
        unavailable = client.get(f"/c/{published.public_slug}")
    assert unavailable.status_code == 404


def test_public_chat_uses_private_pairs_server_side_without_exposing_them(
    tmp_path: Path,
) -> None:
    repository, _, service, record = _service(tmp_path)
    published = service.publish(
        OWNER,
        record.capsule_id,
        PublishSelection(
            include_summary=True,
            include_descriptors=True,
            include_dimensions=False,
            include_card=True,
        ),
        confirmed=True,
    )
    gateway = PublicChatSteeringGateway()
    app = create_app(
        Settings(
            app_env="test",
            modal_available=True,
            capsule_data_dir=str(tmp_path),
            public_base_url="https://capsules.example",
            quota_public_chat_daily=1,
        ),
        repository=repository,
        steering_gateway=gateway,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/c/{published.public_slug}/chat",
            json={"message": "What should we test first?"},
        )
        exhausted = client.post(
            f"/c/{published.public_slug}/chat",
            json={"message": "One more question"},
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "Test the smallest reversible version first."
    assert PRIVATE_SENTINEL not in response.text
    assert OWNER.user_id not in response.text
    assert exhausted.status_code == 429
    assert len(gateway.calls) == 1
    assert "reply only in English" in gateway.calls[0]["prompt"]
    assert gateway.calls[0]["prompt"].endswith("Visitor message:\nWhat should we test first?")
    assert gateway.calls[0]["pairs"] == record.exemplar_pairs
    assert gateway.calls[0]["strength"] == 0.85


def test_public_challenge_shuffles_baseline_and_steered_without_private_data(
    tmp_path: Path,
) -> None:
    repository, _, service, record = _service(tmp_path)
    published = service.publish(
        OWNER,
        record.capsule_id,
        PublishSelection(include_card=True),
        confirmed=True,
    )
    gateway = PublicChatSteeringGateway()
    app = create_app(
        Settings(
            app_env="test",
            modal_available=True,
            capsule_data_dir=str(tmp_path),
            public_base_url="https://capsules.example",
            quota_public_chat_daily=1,
        ),
        repository=repository,
        steering_gateway=gateway,
    )

    with TestClient(app) as client:
        response = client.post(f"/c/{published.public_slug}/challenge")
        payload = response.json()
        steered_index = payload["answers"].index("Test the smallest reversible version first.")
        guess = client.post(
            f"/c/{published.public_slug}/challenge/guess",
            json={
                "challenge_id": payload["challenge_id"],
                "guess": steered_index,
            },
        )
        replay = client.post(
            f"/c/{published.public_slug}/challenge/guess",
            json={
                "challenge_id": payload["challenge_id"],
                "guess": steered_index,
            },
        )

    assert response.status_code == 200
    assert set(payload["answers"]) == {
        "A generic answer.",
        "Test the smallest reversible version first.",
    }
    assert "steered_index" not in payload
    assert "small reversible idea" in payload["prompt"]
    assert PRIVATE_SENTINEL not in response.text
    assert OWNER.user_id not in response.text
    assert gateway.calls[0]["pairs"] == record.exemplar_pairs
    assert guess.status_code == 200
    assert guess.json() == {"correct": True, "steered_index": steered_index}
    assert replay.status_code == 404
