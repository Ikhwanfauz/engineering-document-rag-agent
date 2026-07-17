"""Validate the Version 0 project skeleton without third-party dependencies."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "api",
    "dashboard",
    "src",
    "evaluation",
    "database",
    "data/manuals",
    "data/processed",
    "data/vector_store",
    "results/evaluations",
    "results/logs",
    "tests",
    "docs",
    "scripts",
)

REQUIRED_FILES = (
    "README.md",
    "environment.yml",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "docs/project_scope.md",
    "docs/architecture.md",
    "docs/version_roadmap.md",
    "docs/decision_log.md",
    "docs/personal_tracking.md",
)


def find_missing_paths() -> list[str]:
    """Return required Version 0 paths that are currently missing."""
    missing: list[str] = []

    for relative_path in REQUIRED_DIRECTORIES:
        if not (PROJECT_ROOT / relative_path).is_dir():
            missing.append(f"directory: {relative_path}")

    for relative_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative_path).is_file():
            missing.append(f"file: {relative_path}")

    return missing


def main() -> int:
    """Print a clear setup result and return a shell-friendly status code."""
    missing = find_missing_paths()
    if missing:
        print("Version 0 setup check failed. Missing paths:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Version 0 setup check passed.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Validated {len(REQUIRED_DIRECTORIES)} directories and {len(REQUIRED_FILES)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

