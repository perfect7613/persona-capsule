"""Hugging Face Space entry point."""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from persona_capsule.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
