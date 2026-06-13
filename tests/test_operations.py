from pathlib import Path

import pytest

from persona_capsule.operations import (
    DailyQuotaManager,
    FeatureDisabledError,
    FeatureFlags,
    OperationsGuard,
    QuotaExceededError,
    SafeTelemetry,
)


def test_feature_flags_and_daily_quota_are_enforced() -> None:
    guard = OperationsGuard(
        FeatureFlags(voice=False),
        DailyQuotaManager({"battle": 1, "voice": 5}),
        SafeTelemetry(),
    )

    assert guard.require("hf:alice", "battle") == 0
    with pytest.raises(QuotaExceededError):
        guard.require("hf:alice", "battle")
    with pytest.raises(FeatureDisabledError):
        guard.require("hf:alice", "voice")


def test_telemetry_whitelists_operational_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    telemetry = SafeTelemetry(path)

    event = telemetry.record(
        "battle.completed",
        {
            "feature": "battle",
            "outcome": "success",
            "raw_message": "private text",
            "audio": b"private audio",
            "secret": "token",
        },
    )

    assert event["metadata"] == {"feature": "battle", "outcome": "success"}
    serialized = path.read_text(encoding="utf-8")
    assert "private text" not in serialized
    assert "private audio" not in serialized
    assert "token" not in serialized
