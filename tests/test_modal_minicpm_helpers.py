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
