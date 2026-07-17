"""Tests for the Version 0 project structure."""

from scripts.check_setup import find_missing_paths


def test_required_version_zero_paths_exist() -> None:
    assert find_missing_paths() == []

