"""Resumable orchestration for opt-in Modal Deep Capsule jobs."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from persona_capsule.identity import Principal
from persona_capsule.library import CapsuleLibrary
from persona_capsule.repository import CapsuleRecord

TERMINAL_STATES = {"completed", "failed", "cancelled", "timeout"}


class DeepTrainingGateway(Protocol):
    def spawn(self, payload: dict[str, Any]) -> str: ...

    def poll(self, call_id: str) -> dict[str, Any]: ...

    def cancel(self, call_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DeepTrainingEstimate:
    estimated_minutes: int
    estimated_modal_credits: float
    visual_lora: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimated_minutes": self.estimated_minutes,
            "estimated_modal_credits": self.estimated_modal_credits,
            "visual_lora": self.visual_lora,
        }


def estimate_deep_training(*, visual_lora: bool) -> DeepTrainingEstimate:
    return DeepTrainingEstimate(
        estimated_minutes=35,
        estimated_modal_credits=8.0,
        visual_lora=visual_lora,
    )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class DeepCapsuleService:
    def __init__(
        self,
        capsule_library: CapsuleLibrary,
        gateway: DeepTrainingGateway,
    ) -> None:
        self._capsule_library = capsule_library
        self._gateway = gateway

    def start(
        self,
        principal: Principal | None,
        capsule_id: str,
        *,
        idempotency_key: str,
        visual_lora: bool,
        confirmed: bool,
    ) -> CapsuleRecord:
        if principal is None:
            raise PermissionError("Hugging Face login required")
        if not confirmed:
            raise ValueError("Confirm the compute estimate before starting Deep Capsule.")
        key = idempotency_key.strip()
        if not 8 <= len(key) <= 128:
            raise ValueError("Idempotency key must contain between 8 and 128 characters.")
        record = self._capsule_library.get_capsule(principal, capsule_id)
        if not record.exemplar_pairs:
            raise ValueError("Deep Capsule requires approved private steering pairs.")
        existing = record.deep_training or {}
        if existing.get("idempotency_key") == key:
            return record
        if existing.get("status") not in {None, *TERMINAL_STATES}:
            raise ValueError("This capsule already has an active Deep Capsule job.")

        estimate = estimate_deep_training(visual_lora=visual_lora)
        payload = {
            "schema_version": "persona-deep-v1",
            "owner_namespace": principal.user_id,
            "capsule_id": record.capsule_id,
            "capsule_version": record.source_fingerprint,
            "profile": record.style_profile.as_dict(include_private_evidence=False)
            if record.style_profile
            else {},
            "training_pairs": [pair.as_dict() for pair in record.exemplar_pairs],
            "visual_lora": visual_lora,
            "idempotency_key": key,
        }
        call_id = self._gateway.spawn(payload)
        job = {
            "status": "queued",
            "call_id": call_id,
            "idempotency_key": key,
            "visual_lora": visual_lora,
            "estimate": estimate.as_dict(),
            "started_at": _now(),
            "updated_at": _now(),
            "progress": 0,
            "quick_capsule_status": record.status,
        }
        return self._capsule_library.save_capsule(
            principal,
            replace(record, deep_training=job),
        )

    def poll(
        self,
        principal: Principal | None,
        capsule_id: str,
    ) -> CapsuleRecord:
        record = self._capsule_library.get_capsule(principal, capsule_id)
        job = dict(record.deep_training or {})
        call_id = str(job.get("call_id", ""))
        if not call_id:
            raise ValueError("This capsule has no Deep Capsule job.")
        if job.get("status") in TERMINAL_STATES:
            return record
        provider = self._gateway.poll(call_id)
        status = str(provider.get("status", "running"))
        if status not in {"queued", "running", *TERMINAL_STATES}:
            raise ValueError("Deep Capsule provider returned an invalid status.")
        job.update(
            {
                "status": status,
                "progress": max(0, min(100, int(provider.get("progress", job.get("progress", 0))))),
                "updated_at": _now(),
                "message": str(provider.get("message", ""))[:500],
            }
        )
        if status == "completed":
            evaluation = dict(provider.get("evaluation", {}))
            quality_gain = float(evaluation.get("quality_gain", 0))
            memorization = float(evaluation.get("memorization_score", 1))
            passed = quality_gain >= 0.05 and memorization <= 0.2
            job["evaluation"] = evaluation
            job["visual_lora_result"] = dict(provider.get("visual_lora", {}))
            job["artifact_attached"] = passed
            if passed:
                artifact = dict(provider.get("artifact", {}))
                if not artifact.get("repo_id") or not artifact.get("revision"):
                    raise ValueError("Passing Deep Capsule result omitted its model artifact.")
                job["artifact"] = artifact
        elif status in TERMINAL_STATES:
            job["artifact_attached"] = False
        return self._capsule_library.save_capsule(
            principal,
            replace(record, deep_training=job),
        )

    def cancel(
        self,
        principal: Principal | None,
        capsule_id: str,
    ) -> CapsuleRecord:
        record = self._capsule_library.get_capsule(principal, capsule_id)
        job = dict(record.deep_training or {})
        call_id = str(job.get("call_id", ""))
        if not call_id:
            raise ValueError("This capsule has no Deep Capsule job.")
        if job.get("status") not in TERMINAL_STATES:
            self._gateway.cancel(call_id)
            job.update({"status": "cancelled", "updated_at": _now(), "artifact_attached": False})
        return self._capsule_library.save_capsule(
            principal,
            replace(record, deep_training=job),
        )
