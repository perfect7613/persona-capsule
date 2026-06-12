from persona_capsule.config import Settings


def test_settings_report_provider_availability_without_values() -> None:
    settings = Settings.from_env(
        {
            "APP_ENV": "test",
            "HF_TOKEN": "hf-test-value",
            "MODAL_TOKEN_ID": "modal-id",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "ELEVENLABS_API_KEY": "",
        }
    )

    assert settings.app_env == "test"
    assert settings.providers == {
        "hugging_face": True,
        "modal": True,
        "elevenlabs": False,
    }
    assert "hf-test-value" not in repr(settings)
    assert "modal-secret" not in repr(settings)


def test_modal_requires_both_token_parts() -> None:
    settings = Settings.from_env({"MODAL_TOKEN_ID": "modal-id"})

    assert settings.modal_available is False


def test_local_identity_is_only_allowed_in_development_or_test() -> None:
    production = Settings.from_env(
        {
            "APP_ENV": "production",
            "PERSONA_LOCAL_IDENTITY": "true",
            "PERSONA_LOCAL_HF_USERNAME": "owner",
        }
    )
    development = Settings.from_env(
        {
            "APP_ENV": "development",
            "PERSONA_LOCAL_IDENTITY": "true",
            "PERSONA_LOCAL_HF_USERNAME": "owner",
        }
    )

    assert production.local_identity_allowed is False
    assert development.local_identity_allowed is True


def test_oauth_ui_is_available_in_space_or_with_local_hf_token() -> None:
    assert Settings(space_environment=True).oauth_ui_available is True
    assert Settings(local_oauth_available=True).oauth_ui_available is True
    assert Settings(hugging_face_available=True).oauth_ui_available is False
    assert Settings().oauth_ui_available is False
