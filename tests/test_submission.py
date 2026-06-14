import importlib.util
from pathlib import Path


def _load_checker():
    path = Path(__file__).parents[1] / "scripts" / "check_submission.py"
    spec = importlib.util.spec_from_file_location("check_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_deployer():
    path = Path(__file__).parents[1] / "scripts" / "deploy_space.py"
    spec = importlib.util.spec_from_file_location("deploy_space", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_submission_checks_pass_before_owner_links() -> None:
    checker = _load_checker()
    assert checker.collect_errors(strict=False) == []


def test_strict_submission_requires_owner_reviewed_links() -> None:
    checker = _load_checker()
    errors = checker.collect_errors(strict=True)
    assert any("demo video" in error for error in errors)
    assert any("social post" in error for error in errors)


def test_space_deployment_plan_uses_official_org_and_secret_names_only() -> None:
    deployer = _load_deployer()
    plan = deployer.deployment_plan()
    assert plan["space_id"] == "build-small-hackathon/persona-capsule"
    assert plan["hardware"] == "cpu-basic"
    assert plan["oauth"] is True
    assert "HF_TOKEN" in plan["secret_names"]
    assert not any("sk_" in value or "hf_" in value for value in plan["secret_names"])
