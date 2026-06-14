"""Modal client for FLUX.2 Klein card-art generation."""

from persona_capsule.card import ArtResult, CardPrompt

FLUX_APP_NAME = "persona-capsule-flux"
FLUX_CLASS_NAME = "FluxCardRuntime"


class ModalFluxArtGateway:
    def generate(self, prompt: CardPrompt) -> ArtResult:
        import modal

        runtime = modal.Cls.from_name(FLUX_APP_NAME, FLUX_CLASS_NAME)()
        result = runtime.generate.remote(
            prompt=prompt.text,
            seed=prompt.seed,
            width=768,
            height=768,
        )
        return ArtResult(
            png_bytes=result["png_bytes"],
            provider=result["provider"],
            model_id=result["model_id"],
            model_revision=result["model_revision"],
        )
