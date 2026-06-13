"""Modal asynchronous-job gateway for Deep Capsule training."""

from typing import Any

import modal

APP_NAME = "persona-capsule-deep"
FUNCTION_NAME = "train_deep_capsule"


class ModalDeepTrainingGateway:
    def spawn(self, payload: dict[str, Any]) -> str:
        function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
        call = function.spawn(payload)
        return call.object_id

    def poll(self, call_id: str) -> dict[str, Any]:
        call = modal.FunctionCall.from_id(call_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return {
                "status": "running",
                "progress": 35,
                "message": "Modal training is still running. Poll again to resume.",
            }
        except modal.exception.OutputExpiredError:
            return {
                "status": "timeout",
                "progress": 0,
                "message": "Modal result retention expired before the job was collected.",
            }
        if not isinstance(result, dict):
            raise RuntimeError("Deep Capsule job returned an invalid result.")
        return result

    def cancel(self, call_id: str) -> None:
        modal.FunctionCall.from_id(call_id).cancel()
