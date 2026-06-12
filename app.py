"""Hugging Face Space entry point."""

import uvicorn

from persona_capsule.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
