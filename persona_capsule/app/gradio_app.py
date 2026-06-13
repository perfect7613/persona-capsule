"""Gradio application with visual LoRA training integration."""

from __future__ import annotations

import json

import gradio as gr

from persona_capsule.config import Settings, ensure_data_dirs
from persona_capsule.models.capsule import CapsuleRecord, CreationMode
from persona_capsule.models.profile import StyleProfile, StyleTrait
from persona_capsule.services.storage import CapsuleRepository
from persona_capsule.services.training_orchestrator import TrainingOrchestrator


def _demo_capsule(settings: Settings) -> CapsuleRecord:
    profile = StyleProfile(
        summary="Demo capsule for local testing and training workflow previews.",
        tone="warm and conversational",
        vocabulary="casual and playful",
        cadence="medium-length sentences",
        traits=[
            StyleTrait("openness", 0.62, "imaginative", ["Let's try something new."]),
            StyleTrait("extraversion", 0.58, "outgoing", ["This is exciting!"]),
        ],
        signature_phrases=["honestly", "let's go"],
        palette="warm amber and deep teal",
        visual_energy="balanced",
        visual_symbols=["chat bubble", "spark"],
    )
    from persona_capsule.models.capsule import ExemplarPair

    return CapsuleRecord.new(
        owner_id=settings.dev_user_id,
        display_name="Demo Capsule",
        profile=profile,
        exemplars=[
            ExemplarPair(
                style_example="Honestly, I love building playful tools that feel personal.",
                neutral_contrast="They enjoy building useful tools.",
            )
        ],
        creation_mode=CreationMode.DEMO,
    )


def build_app(settings: Settings) -> gr.Blocks:
    ensure_data_dirs(settings)
    repo = CapsuleRepository(settings)
    trainer = TrainingOrchestrator(settings)

    demo = _demo_capsule(settings)
    if repo.get(demo.id) is None:
        repo.save(demo)

    def list_capsule_choices() -> list[tuple[str, str]]:
        capsules = repo.list_for_owner(settings.dev_user_id)
        if not capsules:
            capsules = [demo]
        return [(f"{c.display_name} ({c.id[:8]})", c.id) for c in capsules]

    def preview_training_config(
        capsule_id: str,
        dataset_path: str,
        steps: int,
        trigger_word: str,
    ) -> str:
        capsule = repo.get(capsule_id) or demo
        config_dict = trainer.build_training_config(
            capsule,
            dataset_path or settings.modal_dataset_path or "/dataset/dataset/PicS/PicS",
            steps=int(steps),
            trigger_word=trigger_word or None,
        )
        return json.dumps(config_dict, indent=2)

    def submit_training(
        capsule_id: str,
        dataset_path: str,
        steps: int,
        trigger_word: str,
        dry_run: bool,
    ) -> str:
        capsule = repo.get(capsule_id) or demo
        try:
            job = trainer.submit_visual_lora_training(
                capsule,
                owner_id=settings.dev_user_id,
                dataset_folder_path=dataset_path or None,
                steps=int(steps),
                trigger_word=trigger_word or None,
                dry_run=dry_run,
            )
            capsule = trainer.attach_latest_job_to_capsule(capsule)
            capsule.dataset_folder_path = dataset_path or settings.modal_dataset_path
            repo.save(capsule)
            return json.dumps(job.to_dict(), indent=2)
        except Exception as exc:  # noqa: BLE001 - surface to UI
            return json.dumps({"error": str(exc)}, indent=2)

    def list_training_jobs(capsule_id: str) -> str:
        jobs = trainer.list_jobs_for_capsule(capsule_id)
        return json.dumps([job.to_dict() for job in jobs], indent=2)

    with gr.Blocks(title="Persona Capsule") as app:
        gr.Markdown(
            "# Persona Capsule\n"
            "Create capsules and trigger FLUX.2 Klein visual LoRA training through "
            "the restored ai-toolkit + Modal pipeline."
        )
        with gr.Tab("Visual LoRA Training"):
            capsule_dropdown = gr.Dropdown(
                label="Capsule",
                choices=list_capsule_choices(),
                value=demo.id,
            )
            dataset_path = gr.Textbox(
                label="Modal dataset folder path",
                placeholder="/dataset/dataset/PicS/PicS",
                value=settings.modal_dataset_path or "/dataset/dataset/PicS/PicS",
            )
            steps = gr.Slider(label="Training steps", minimum=250, maximum=4000, value=1800, step=50)
            trigger_word = gr.Textbox(label="Trigger word (optional)", placeholder="Auto-generated from capsule")
            dry_run = gr.Checkbox(
                label="Dry run (generate config + command only)",
                value=settings.use_mock_providers,
            )
            preview_btn = gr.Button("Preview training config")
            submit_btn = gr.Button("Submit visual LoRA training", variant="primary")
            refresh_btn = gr.Button("Refresh job history")
            config_output = gr.Code(label="Training config / job result", language="json")
            preview_btn.click(
                preview_training_config,
                inputs=[capsule_dropdown, dataset_path, steps, trigger_word],
                outputs=config_output,
            )
            submit_btn.click(
                submit_training,
                inputs=[capsule_dropdown, dataset_path, steps, trigger_word, dry_run],
                outputs=config_output,
            )
            refresh_btn.click(list_training_jobs, inputs=[capsule_dropdown], outputs=config_output)

        with gr.Tab("Instructions"):
            gr.Markdown(
                """
### Modal training prerequisites
1. Create Modal volumes: `my-dataset` (images + captions) and `flux-lora-models` (outputs).
2. Set `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and optionally `MODAL_DATASET_PATH` in `.env`.
3. Install Modal CLI: `pip install modal`.
4. Deploy/run from repo root:
   `modal run run_modal.py --config /root/ai-toolkit/artifacts/training_configs/<job>.yaml`

### Local dry run
- Keep `PERSONA_CAPSULE_MODE=local` or enable **Dry run** to generate configs without calling Modal.

### Restored ai-toolkit paths
- `run_modal.py`, `run.py`, `toolkit/`, `jobs/`, `extensions_built_in/`, `config/examples/persona_capsule.yaml`
                """
            )

    return app
