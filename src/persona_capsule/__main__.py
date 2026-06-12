"""Local Persona Capsule server entry point."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "persona_capsule.app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
