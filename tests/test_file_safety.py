"""Tests for uploaded-document filename and path safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.file_safety import (
    UnsafeFilePathError,
    resolve_safe_file_path,
    validate_safe_filename,
)


@pytest.mark.parametrize(
    "filename",
    [
        "manual.pdf",
        "service_manual-v2.pdf",
        "Robot Service Manual.pdf",
        "manual_ä.pdf",
    ],
)
def test_safe_filenames_are_preserved(filename: str) -> None:
    assert validate_safe_filename(filename) == filename


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        ".",
        "..",
        "../manual.pdf",
        "..\\manual.pdf",
        "documents/manual.pdf",
        "documents\\manual.pdf",
        "/absolute/manual.pdf",
        "C:\\manual.pdf",
        "C:manual.pdf",
        " manual.pdf",
        "manual.pdf ",
        "manual.pdf.",
        "manual\x00.pdf",
        "manual\n.pdf",
    ],
)
def test_unsafe_filenames_are_rejected(filename: str) -> None:
    with pytest.raises(UnsafeFilePathError):
        validate_safe_filename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "CON",
        "CON.pdf",
        "prn.PDF",
        "AUX.txt",
        "nul.pdf",
        "COM1.pdf",
        "com9.txt",
        "LPT1.pdf",
        "lpt9.txt",
    ],
)
def test_windows_reserved_names_are_rejected(filename: str) -> None:
    with pytest.raises(
        UnsafeFilePathError,
        match="reserved Windows device name",
    ):
        validate_safe_filename(filename)


def test_safe_file_path_resolves_inside_base_directory(
    tmp_path: Path,
) -> None:
    upload_directory = tmp_path / "uploads"
    upload_directory.mkdir()

    result = resolve_safe_file_path(
        upload_directory,
        "service-manual.pdf",
    )

    assert result == (upload_directory / "service-manual.pdf").resolve()
    assert result.parent == upload_directory.resolve()


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.pdf",
        "..\\outside.pdf",
        "nested/manual.pdf",
        "C:\\outside.pdf",
    ],
)
def test_unsafe_paths_cannot_be_resolved(
    tmp_path: Path,
    filename: str,
) -> None:
    upload_directory = tmp_path / "uploads"
    upload_directory.mkdir()

    with pytest.raises(UnsafeFilePathError):
        resolve_safe_file_path(upload_directory, filename)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    upload_directory = tmp_path / "uploads"
    upload_directory.mkdir()

    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"outside")

    symlink_path = upload_directory / "linked.pdf"

    try:
        symlink_path.symlink_to(outside_file)
    except OSError:
        pytest.skip("File symlinks are unavailable on this system")

    with pytest.raises(
        UnsafeFilePathError,
        match="outside the allowed directory",
    ):
        resolve_safe_file_path(upload_directory, "linked.pdf")