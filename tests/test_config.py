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
