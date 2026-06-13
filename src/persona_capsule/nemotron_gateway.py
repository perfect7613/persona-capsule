"""Modal gateway for the pinned Nemotron battle judge."""

from time import sleep
from typing import Any

import modal

APP_NAME = "persona-capsule-nemotron"
CLASS_NAME = "NemotronBattleRuntime"


class ModalNemotronGateway:
    def __init__(self, *, attempts: int = 2) -> None:
        self._attempts = max(1, int(attempts))

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime = modal.Cls.from_name(APP_NAME, CLASS_NAME)()
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                return runtime.judge.remote(payload=payload)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self._attempts:
                    sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            "Nemotron judging failed after bounded retries. The capsule responses remain intact."
        ) from last_error
