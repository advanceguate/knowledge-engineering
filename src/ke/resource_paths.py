"""Resolve project resources in editable checkouts and installed wheels."""

from __future__ import annotations

from pathlib import Path


def resource_path(group: str, name: str) -> Path:
    """Return a packaged config, prompt, or shape without allowing traversal."""

    if group not in {"config", "prompts", "shapes"}:
        raise ValueError(f"unknown resource group: {group!r}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe resource name: {name!r}")

    package_candidate = Path(__file__).resolve().parent / "_resources" / group / relative
    if package_candidate.is_file():
        return package_candidate

    checkout_candidate = Path(__file__).resolve().parents[2] / group / relative
    if checkout_candidate.is_file():
        return checkout_candidate
    raise FileNotFoundError(f"packaged resource does not exist: {group}/{name}")


def read_resource_text(group: str, name: str) -> str:
    return resource_path(group, name).read_text(encoding="utf-8")
