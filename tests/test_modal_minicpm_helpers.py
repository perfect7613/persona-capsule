import importlib.util
from pathlib import Path


def _load_runtime_module():
    path = Path(__file__).parents[1] / "modal_minicpm.py"
    spec = importlib.util.spec_from_file_location("modal_minicpm_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_legacy_pair_is_repaired() -> None:
    runtime = _load_runtime_module()

    neutral, repaired = runtime.ensure_distinct_contrast(
        "The current plan has too many moving pieces.",
        "The current plan has too many moving pieces.",
    )

    assert repaired is True
    assert neutral != "The current plan has too many moving pieces."


def test_existing_distinct_contrast_is_preserved() -> None:
    runtime = _load_runtime_module()

    neutral, repaired = runtime.ensure_distinct_contrast(
        "Good progress! Let us test it now.",
        "Test it now.",
    )

    assert repaired is False
    assert neutral == "Test it now."


def test_content_span_finds_last_exact_token_sequence() -> None:
    runtime = _load_runtime_module()

    assert runtime._find_content_span([1, 2, 3, 2, 3, 4], [2, 3]) == (3, 5)
    assert runtime._find_content_span([1, 2, 3], [8]) is None


def test_generation_prompt_preserves_request_language() -> None:
    runtime = _load_runtime_module()

    messages = runtime._generation_messages(
        "Reply as Shakespeare: Should we test the risky assumption first?"
    )

    assert messages[0]["role"] == "system"
    assert "answer only in English" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "Reply as Shakespeare: Should we test the risky assumption first?",
    }


def test_vector_calibration_compresses_extreme_magnitudes() -> None:
    runtime = _load_runtime_module()

    assert runtime._calibrated_vector_scale(0.01) == 0.5
    assert runtime._calibrated_vector_scale(4.0) == 4.0
    assert runtime._calibrated_vector_scale(100.0) == 12.0


def test_steering_changes_all_active_tokens_without_mutating_input() -> None:
    import pytest

    torch = pytest.importorskip("torch")

    runtime = _load_runtime_module()
    hidden_states = torch.zeros((1, 3, 2))
    direction = torch.tensor([2.0, -1.0])

    steered = runtime._steer_hidden_states(hidden_states, direction, 0.5)

    assert torch.equal(
        steered,
        torch.tensor([[[1.0, -0.5], [1.0, -0.5], [1.0, -0.5]]]),
    )
    assert torch.equal(hidden_states, torch.zeros((1, 3, 2)))
