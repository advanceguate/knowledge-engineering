"""Atomic filesystem artifact store with content-addressed cache entries."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from ke.ids import canonical_json, content_hash, sha256_file


class ArtifactStore:
    def __init__(self, root: str | Path = "workspace") -> None:
        self.root = Path(root).expanduser().resolve()
        self.sources = self.root / "sources"
        self.ontologies = self.root / "ontologies"
        self.syntheses = self.root / "syntheses"
        self.cache = self.root / "cache"
        self.runs = self.root / "runs"
        for directory in (
            self.sources,
            self.ontologies,
            self.syntheses,
            self.cache,
            self.runs,
        ):
            if directory.is_symlink():
                raise ValueError(f"managed artifact directory cannot be a symlink: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

    def source_dir(self, source_id: str) -> Path:
        return self._safe_child(self.sources, source_id)

    def ontology_dir(self, ontology_id: str) -> Path:
        return self._safe_child(self.ontologies, ontology_id)

    def synthesis_dir(self, frame_id: str) -> Path:
        return self._safe_child(self.syntheses, frame_id)

    @staticmethod
    def _safe_child(parent: Path, name: str) -> Path:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"unsafe artifact identifier: {name!r}")
        resolved_parent = parent.resolve()
        candidate = parent / name
        if candidate.is_symlink():
            raise ValueError(f"managed artifact path cannot be a symlink: {candidate}")
        resolved_candidate = candidate.resolve(strict=False)
        if resolved_candidate.parent != resolved_parent:
            raise ValueError(f"artifact path escapes its managed directory: {candidate}")
        return candidate

    def _managed_path(self, path: str | Path) -> Path:
        """Return an in-store path after rejecting escapes and symlinked components."""

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).absolute()
        root = self.root.resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path is outside the managed store: {candidate}") from exc
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"managed artifact path cannot contain symlinks: {cursor}")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes the managed store: {candidate}") from exc
        return candidate

    def _atomic_bytes(self, path: Path, value: bytes) -> Path:
        path = self._managed_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return path

    def write_bytes(self, path: str | Path, value: bytes) -> Path:
        return self._atomic_bytes(Path(path), value)

    def write_text(self, path: str | Path, value: str) -> Path:
        return self._atomic_bytes(Path(path), value.encode("utf-8"))

    def write_json(self, path: str | Path, value: Any, *, indent: int = 2) -> Path:
        if indent == 2:
            data = (
                json.dumps(
                    value.model_dump(mode="json") if hasattr(value, "model_dump") else value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
                + b"\n"
            )
        else:
            data = canonical_json(value)
        return self._atomic_bytes(Path(path), data)

    def write_jsonl(self, path: str | Path, values: Iterable[Any]) -> Path:
        lines = [canonical_json(value) for value in values]
        return self._atomic_bytes(Path(path), b"\n".join(lines) + (b"\n" if lines else b""))

    def write_yaml(self, path: str | Path, value: Any) -> Path:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        data = yaml.safe_dump(value, allow_unicode=True, sort_keys=False).encode("utf-8")
        return self._atomic_bytes(Path(path), data)

    def read_json(self, path: str | Path) -> Any:
        return json.loads(self._managed_path(path).read_text(encoding="utf-8"))

    def read_jsonl(self, path: str | Path) -> list[Any]:
        return [
            json.loads(line)
            for line in self._managed_path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def copy_file(self, source: str | Path, destination: str | Path) -> Path:
        source_path = Path(source)
        destination_path = self._managed_path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() and sha256_file(source_path) == sha256_file(destination_path):
            return destination_path
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.", dir=destination_path.parent
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            shutil.copy2(source_path, temporary_path)
            temporary_path.replace(destination_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination_path

    def cache_path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        safe_namespace = self._safe_child(self.cache, namespace)
        safe_namespace.mkdir(parents=True, exist_ok=True)
        safe_key = content_hash(key) if any(char in key for char in "/\\\0") else key
        return safe_namespace / f"{safe_key}{suffix}"

    def get_cached_json(self, namespace: str, key: str) -> Any | None:
        path = self.cache_path(namespace, key)
        if not path.exists():
            return None
        try:
            return self.read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def put_cached_json(self, namespace: str, key: str, value: Any) -> Path:
        return self.write_json(self.cache_path(namespace, key), value)
