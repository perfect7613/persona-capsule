import importlib.util
from pathlib import Path


def _load_runtime_module():
    path = Path(__file__).parents[1] / "modal_nemotron.py"
    spec = importlib.util.spec_from_file_location("modal_nemotron_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_judgment_wraps_flat_candidate_scores() -> None:
    runtime = _load_runtime_module()
    normalized = runtime._normalize_judgment(
        {
            "candidate_a": {
                "style_fidelity": 9,
                "response_quality": 8,
                "instruction_adherence": 10,
                "safety": 10,
                "rationale": "A is direct.",
            },
            "candidate_b": {
                "style_fidelity": 6,
                "response_quality": 7,
                "instruction_adherence": 8,
                "safety": 10,
                "rationale": "B is vague.",
            },
        }
    )

    assert normalized["scores"]["candidate_a"]["style_fidelity"] == 9
    assert normalized["scores"]["candidate_b"]["response_quality"] == 7
    assert normalized["rationale"] == "candidate_a: A is direct. candidate_b: B is vague."


def test_normalize_judgment_preserves_schema_compliant_scores() -> None:
    payload = {
        "scores": {
            "candidate_a": {
                "style_fidelity": 9,
                "response_quality": 8,
                "instruction_adherence": 10,
                "safety": 10,
            },
            "candidate_b": {
                "style_fidelity": 6,
                "response_quality": 7,
                "instruction_adherence": 8,
                "safety": 10,
            },
        },
        "rationale": "A is better.",
    }

    runtime = _load_runtime_module()
    assert runtime._normalize_judgment(payload) is payload
