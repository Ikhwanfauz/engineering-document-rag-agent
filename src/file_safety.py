"""Filename and path-safety checks for uploaded documents."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafeFilePathError(ValueError):
    """Raised when a filename or resolved path is unsafe."""


_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


def validate_safe_filename(filename: str) -> str:
    """Return an unchanged filename after verifying that it is safe."""
    if not filename or not filename.strip():
        raise UnsafeFilePathError("Filename cannot be empty")

    if filename != filename.strip():
        raise UnsafeFilePathError(
            "Filename cannot begin or end with whitespace"
        )

    if filename in {".", ".."}:
        raise UnsafeFilePathError("Filename cannot be a relative path marker")

    if "/" in filename or "\\" in filename:
        raise UnsafeFilePathError(
            "Filename must not contain directory components"
        )

    if (
        PurePosixPath(filename).name != filename
        or PureWindowsPath(filename).name != filename
    ):
        raise UnsafeFilePathError(
            "Filename must not contain a path or drive prefix"
        )

    if ":" in filename:
        raise UnsafeFilePathError("Filename must not contain a colon")

    if any(ord(character) < 32 for character in filename):
        raise UnsafeFilePathError(
            "Filename must not contain control characters"
        )

    if filename.endswith((".", " ")):
        raise UnsafeFilePathError(
            "Filename must not end with a dot or space"
        )

    reserved_candidate = filename.split(".", maxsplit=1)[0].upper()
    if reserved_candidate in _WINDOWS_RESERVED_NAMES:
        raise UnsafeFilePathError(
            "Filename uses a reserved Windows device name"
        )

    return filename


def resolve_safe_file_path(
    base_directory: Path,
    filename: str,
) -> Path:
    """Resolve a filename and ensure it remains inside its base directory."""
    safe_filename = validate_safe_filename(filename)
    resolved_base = base_directory.resolve()
    resolved_path = (resolved_base / safe_filename).resolve()

    try:
        resolved_path.relative_to(resolved_base)
    except ValueError as error:
        raise UnsafeFilePathError(
            "Resolved file path is outside the allowed directory"
        ) from error

    return resolved_path