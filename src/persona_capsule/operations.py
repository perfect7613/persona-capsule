"""Feature switches, per-user quotas, and privacy-safe operational events."""

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from threading import RLock
from typing import Any

SAFE_EVENT_FIELDS = {
    "capsule_id",
    "duration_ms",
    "feature",
    "outcome",
    "provider",
    "status",
}


class FeatureDisabledError(RuntimeError):
    """Raised when an administrator disables an optional provider feature."""


class QuotaExceededError(RuntimeError):
    """Raised before a provider call would exceed the configured user quota."""


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    creation: bool = True
    steering: bool = True
    card: bool = True
    voice: bool = True
    fusion: bool = True
    battle: bool = True
    deep_training: bool = True

    def enabled(self, feature: str) -> bool:
        if not hasattr(self, feature):
            raise KeyError(f"Unknown feature: {feature}")
        return bool(getattr(self, feature))

    def as_dict(self) -> dict[str, bool]:
        return {
            name: bool(getattr(self, name))
            for name in (
                "creation",
                "steering",
                "card",
                "voice",
                "fusion",
                "battle",
                "deep_training",
            )
        }


class DailyQuotaManager:
    """Process-local daily quotas suited to a bounded hackathon deployment."""

    def __init__(self, limits: Mapping[str, int]) -> None:
        self._limits = {name: max(0, int(limit)) for name, limit in limits.items()}
        self._usage: dict[tuple[date, str, str], int] = defaultdict(int)
        self._lock = RLock()

    def consume(self, owner_id: str, feature: str, amount: int = 1) -> int:
        limit = self._limits.get(feature, 0)
        key = (datetime.now(UTC).date(), owner_id, feature)
        with self._lock:
            next_value = self._usage[key] + max(1, int(amount))
            if limit <= 0 or next_value > limit:
                raise QuotaExceededError(
                    f"Daily {feature.replace('_', ' ')} quota reached. Try again tomorrow."
                )
            self._usage[key] = next_value
            return limit - next_value


class SafeTelemetry:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._events: list[dict[str, Any]] = []
        self._lock = RLock()

    def record(self, event: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        clean_event = "".join(
            character for character in event if character.isalnum() or character in "._-"
        )
        if not clean_event:
            raise ValueError("Telemetry event name is required.")
        safe = {
            key: value
            for key, value in metadata.items()
            if key in SAFE_EVENT_FIELDS and isinstance(value, (bool, float, int, str))
        }
        payload = {
            "event": clean_event,
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "metadata": safe,
        }
        with self._lock:
            self._events.append(payload)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
                self._path.chmod(0o600)
        return payload

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)


class OperationsGuard:
    def __init__(
        self,
        flags: FeatureFlags,
        quotas: DailyQuotaManager,
        telemetry: SafeTelemetry,
    ) -> None:
        self.flags = flags
        self.quotas = quotas
        self.telemetry = telemetry

    def require(self, owner_id: str, feature: str) -> int:
        if not self.flags.enabled(feature):
            raise FeatureDisabledError(
                f"{feature.replace('_', ' ').title()} is temporarily disabled."
            )
        remaining = self.quotas.consume(owner_id, feature)
        self.telemetry.record(
            "quota.consumed",
            {"feature": feature, "outcome": "allowed", "status": str(remaining)},
        )
        return remaining
