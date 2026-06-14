"""Check repository-level hackathon submission requirements."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REQUIRED_TAGS = {
    "track:wood",
    "sponsor:openbmb",
    "sponsor:openai",
    "sponsor:nvidia",
    "sponsor:modal",
}
REQUIRED_FILES = {
    "app.py",
    "CODEX_USAGE.md",
    "PRD.md",
    "docs/DEMO_SCRIPT.md",
    "docs/DEPLOYMENT.md",
    "docs/MANUAL_ACCEPTANCE.md",
    "modal_deep.py",
    "modal_flux.py",
    "modal_minicpm.py",
    "modal_nemotron.py",
}


def collect_errors(*, strict: bool) -> list[str]:
    text = README.read_text(encoding="utf-8")
    errors = []
    for marker in (
        "sdk: gradio",
        "hf_oauth: true",
        "suggested_hardware: cpu-basic",
        "app_file: app.py",
    ):
        if marker not in text:
            errors.append(f"README metadata missing `{marker}`")
    for tag in sorted(REQUIRED_TAGS):
        if f"- {tag}" not in text:
            errors.append(f"README metadata missing `{tag}`")
    for relative in sorted(REQUIRED_FILES):
        if not (ROOT / relative).is_file():
            errors.append(f"Required file missing: {relative}")

    tracked = subprocess.run(
        ["git", "ls-files", ".env", ".env.*"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    unsafe = [path for path in tracked if path != ".env.example"]
    if unsafe:
        errors.append(f"Credential files are tracked: {', '.join(unsafe)}")

    if strict:
        for label in ("Demo video", "Social post"):
            pattern = rf"{label}:\s*\[[^\]]+\]\(https://[^)]+\)"
            if not re.search(pattern, text, flags=re.IGNORECASE):
                errors.append(f"Strict submission requires a reviewed {label.lower()} URL")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    errors = collect_errors(strict=arguments.strict)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Submission repository checks passed.")


if __name__ == "__main__":
    main()
