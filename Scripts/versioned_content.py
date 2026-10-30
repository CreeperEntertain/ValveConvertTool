"""Resolve versioned Valve assets using per-folder history metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple


_COMPOUND_SUFFIXES = (
    ".dx70.vtx",
    ".dx80.vtx",
    ".dx90.vtx",
    ".dx11.vtx",
    ".dx12.vtx",
    ".sw.vtx",
)
_VERSION_RE = re.compile(r"^(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_METADATA_NAME = ".valve-diff-metadata.json"
_SUPPORTED_SCHEMA_VERSION = 1

_METADATA_CACHE: dict[Path, tuple[int, int, Optional[dict[str, Any]]]] = {}
_HASH_CACHE: dict[Path, tuple[int, int, Optional[str]]] = {}
_WARNED_METADATA: set[Path] = set()


def _version_suffix(filename: str) -> str:
    lower = filename.casefold()
    for suffix in _COMPOUND_SUFFIXES:
        if lower.endswith(suffix):
            return filename[-len(suffix):]
    return Path(filename).suffix


def _variant_number(
    container_name: str,
    child_name: str,
) -> Optional[int]:
    """Return N for file.ext/N.ext; otherwise return None."""
    suffix = _version_suffix(container_name)
    if not suffix or not child_name.casefold().endswith(suffix.casefold()):
        return None
    number_text = child_name[:-len(suffix)]
    if not _VERSION_RE.fullmatch(number_text):
        return None
    return int(number_text)


def split_versioned_relative(
    relative_path: str | os.PathLike[str],
) -> Tuple[str, Optional[int]]:
    """Return the logical path and legacy number encoded by its final path pair."""
    text = os.fspath(relative_path).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative asset path: {text}")
    if len(pure.parts) < 2:
        return pure.as_posix(), None
    version = _variant_number(pure.parts[-2], pure.parts[-1])
    if version is None:
        return pure.as_posix(), None
    logical = PurePosixPath(*pure.parts[:-2], pure.parts[-2])
    return logical.as_posix(), version


def _find_variants(container: Path) -> Optional[Dict[int, Path]]:
    """Inspect one numbered-variant container without walking the archive."""
    if not container.is_dir() or container.is_symlink():
        return None
    variants: Dict[int, Path] = {}
    try:
        for child in container.iterdir():
            if not child.is_file() or child.is_symlink():
                continue
            version = _variant_number(container.name, child.name)
            if version is not None:
                variants[version] = child
    except OSError:
        return None
    return variants or None


def choose_version(
    versions: Iterable[int],
    desired_version: Optional[int],
) -> Optional[int]:
    """Choose latest legacy number, or highest number <= the requested one."""
    available = sorted({int(version) for version in versions})
    if not available:
        return None
    if desired_version is None:
        return available[-1]
    eligible = [version for version in available if version <= desired_version]
    return eligible[-1] if eligible else None


def _utc_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _normalized_build(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text or None
    return None


def _normalize_metadata_entry(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    digest = raw.get("sha256")
    if not isinstance(digest, str):
        return None
    digest = digest.strip().removeprefix("sha256:").lower()
    if not _SHA256_RE.fullmatch(digest):
        return None
    version = raw.get("version")
    if version is not None and not isinstance(version, (str, int, float)):
        version = None
    return {
        "sha256": digest,
        "version": version,
        "introduced_at": _utc_datetime(raw.get("introducedAt")),
        "build": _normalized_build(raw.get("build")),
    }


def _load_metadata(path: Path) -> Optional[dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _METADATA_CACHE.get(path)
    if cached and cached[:2] == signature:
        return cached[2]

    document: Optional[dict[str, Any]] = None
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            candidate = json.load(file)
        if not isinstance(candidate, dict):
            raise ValueError("top-level JSON value must be an object")
        schema_version = candidate.get("schemaVersion")
        if schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schemaVersion {schema_version!r}; "
                f"expected {_SUPPORTED_SCHEMA_VERSION}"
            )
        if not isinstance(candidate.get("files"), dict):
            raise ValueError("'files' must be an object")
        document = candidate
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if path not in _WARNED_METADATA:
            print(
                f"[versioning] Ignoring invalid metadata {path}: {exc}",
                flush=True,
            )
            _WARNED_METADATA.add(path)
    _METADATA_CACHE[path] = (signature[0], signature[1], document)
    return document


def _hash_file(path: Path) -> Optional[str]:
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _HASH_CACHE.get(path)
    if cached and cached[:2] == signature:
        return cached[2]
    try:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        result: Optional[str] = digest.hexdigest()
    except OSError:
        result = None
    _HASH_CACHE[path] = (signature[0], signature[1], result)
    return result


def _safe_relative_text(path: str) -> Optional[str]:
    text = str(path).replace("\\", "/").strip()
    if not text or text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return None
    pure = PurePosixPath(text)
    if ".." in pure.parts:
        return None
    normalized = pure.as_posix()
    return "" if normalized == "." else normalized


def _relative_to(path: Path, base: Path) -> Optional[str]:
    try:
        value = path.relative_to(base).as_posix()
    except ValueError:
        return None
    return _safe_relative_text(value)


def _metadata_directories(
    logical_path: Path,
    physical_candidates: list[tuple[Optional[int], Path]],
    search_root: Path,
) -> list[Path]:
    """Return only ancestor directories relevant to this logical asset."""
    root = search_root.resolve()
    starts: list[Path] = [logical_path]
    if physical_candidates:
        starts.extend(path.parent for _, path in physical_candidates)
    else:
        starts.append(logical_path.parent)

    result: list[Path] = []
    seen: set[Path] = set()
    for start in starts:
        try:
            current = start.resolve()
            current.relative_to(root)
        except (OSError, ValueError):
            continue
        while True:
            if current not in seen:
                seen.add(current)
                result.append(current)
            if current == root:
                break
            parent = current.parent
            if parent == current:
                break
            try:
                parent.relative_to(root)
            except ValueError:
                break
            current = parent
    return result


def _history_for_path(
    logical_path: Path,
    physical_candidates: list[tuple[Optional[int], Path]],
    search_root: Path,
) -> tuple[list[dict[str, Any]], bool]:
    """Load the nearest matching path history from ancestor metadata files."""
    root = search_root.resolve()
    logical_root_key = _relative_to(logical_path.resolve(), root)

    for directory in _metadata_directories(
        logical_path,
        physical_candidates,
        root,
    ):
        metadata_path = directory / _METADATA_NAME
        if not metadata_path.is_file() or metadata_path.is_symlink():
            continue
        document = _load_metadata(metadata_path)
        if document is None:
            continue

        files = document["files"]
        aliases: list[str] = []
        relative_logical = _relative_to(logical_path.resolve(), directory)

        for alias in (relative_logical, logical_root_key, logical_path.name):
            if alias and alias.casefold() not in {
                item.casefold() for item in aliases
            }:
                aliases.append(alias)

        matched_values: list[Any] = []
        matched_keys: set[str] = set()
        key_lookup = {
            key.casefold(): key
            for key in files.keys()
            if isinstance(key, str)
        }

        for alias in aliases:
            original = key_lookup.get(alias.casefold())
            if original is not None and original.casefold() not in matched_keys:
                matched_values.append(files[original])
                matched_keys.add(original.casefold())
                break

        if not matched_values:
            child_aliases: list[str] = []
            for _, physical_path in physical_candidates:
                relative_physical = _relative_to(
                    physical_path.resolve(),
                    directory,
                )
                for alias in (relative_physical, physical_path.name):
                    if alias and alias.casefold() not in {
                        item.casefold() for item in child_aliases
                    }:
                        child_aliases.append(alias)

            for alias in child_aliases:
                original = key_lookup.get(alias.casefold())
                if original is not None and original.casefold() not in matched_keys:
                    matched_values.append(files[original])
                    matched_keys.add(original.casefold())

        if matched_values:
            normalized: list[dict[str, Any]] = []
            seen_records: set[tuple[Any, ...]] = set()
            for value in matched_values:
                values = value if isinstance(value, list) else [value]
                for raw in values:
                    entry = _normalize_metadata_entry(raw)
                    if entry is None:
                        continue
                    signature = (
                        entry["sha256"],
                        entry["introduced_at"],
                        entry["build"],
                        entry["version"],
                    )
                    if signature not in seen_records:
                        seen_records.add(signature)
                        normalized.append(entry)
            return normalized, True

        candidate_hashes = {
            path: _hash_file(path)
            for _, path in physical_candidates
        }
        wanted_hashes = {digest for digest in candidate_hashes.values() if digest}
        matches: list[Any] = []
        matching_keys: set[str] = set()

        for key, value in files.items():
            values = value if isinstance(value, list) else [value]
            entries = [_normalize_metadata_entry(raw) for raw in values]
            if any(
                entry and entry["sha256"] in wanted_hashes
                for entry in entries
            ):
                matching_keys.add(str(key).casefold())
                matches.extend(values)

        if len(matching_keys) == 1:
            normalized = [
                entry
                for raw in matches
                if (entry := _normalize_metadata_entry(raw)) is not None
            ]
            return normalized, True

    return [], False


@dataclass(frozen=True)
class VersionSelectionContext:
    """Historical context of the asset being converted."""

    legacy_version: Optional[int] = None
    build: Optional[str] = None
    introduced_at: Optional[datetime] = None
    metadata_used: bool = False
    search_root: Optional[Path] = None

    def __str__(self) -> str:
        if self.build and self.introduced_at:
            return f"build {self.build} at {self.introduced_at.date().isoformat()}"
        if self.build:
            return f"build {self.build}"
        if self.introduced_at:
            return self.introduced_at.date().isoformat()
        if self.legacy_version is not None:
            return str(self.legacy_version)
        return "latest"


def get_selection_context(
    input_root: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    legacy_version: Optional[int],
) -> VersionSelectionContext:
    """Identify a physical input variant from its digest and history record."""
    root = Path(input_root).resolve()
    source = Path(input_path).resolve()
    try:
        relative_input = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Input file is outside the input directory: {input_path}"
        ) from exc

    logical_relative, _ = split_versioned_relative(relative_input)
    logical_path = root.joinpath(*PurePosixPath(logical_relative).parts)
    physical_candidates = _physical_candidates(logical_path)

    if source.is_file() and all(path != source for _, path in physical_candidates):
        physical_candidates.append(
            (_variant_number(source.parent.name, source.name), source)
        )

    history, _ = _history_for_path(
        logical_path,
        physical_candidates,
        root,
    )
    source_hash = _hash_file(source)
    matches = [
        entry for entry in history
        if source_hash and entry["sha256"] == source_hash
    ]

    if matches:
        matches.sort(key=lambda entry: (
            entry["introduced_at"]
            or datetime.max.replace(tzinfo=timezone.utc),
            entry["build"] or "",
        ))
        record = matches[0]
        return VersionSelectionContext(
            legacy_version=legacy_version,
            build=record["build"],
            introduced_at=record["introduced_at"],
            metadata_used=True,
            search_root=root,
        )

    return VersionSelectionContext(
        legacy_version=legacy_version,
        metadata_used=False,
        search_root=root,
    )


def _physical_candidates(
    logical_path: Path,
) -> list[tuple[Optional[int], Path]]:
    if logical_path.is_file() and not logical_path.is_symlink():
        return [(None, logical_path)]
    variants = _find_variants(logical_path)
    if not variants:
        return []
    return [(number, path) for number, path in sorted(variants.items())]


def _metadata_candidates(
    root: Path,
    logical_path: Path,
    context: Optional[VersionSelectionContext],
) -> tuple[list[tuple[Optional[int], Path, dict[str, Any]]], bool]:
    candidates = _physical_candidates(logical_path)
    if not candidates:
        return [], False

    search_root = (
        context.search_root.resolve()
        if context is not None and context.search_root is not None
        else root.resolve()
    )
    history, metadata_present = _history_for_path(
        logical_path,
        candidates,
        search_root,
    )
    if not metadata_present or not history:
        return [], False

    by_hash: dict[str, list[tuple[Optional[int], Path]]] = {}
    for number, path in candidates:
        digest = _hash_file(path)
        if digest:
            by_hash.setdefault(digest, []).append((number, path))

    mapped: list[tuple[Optional[int], Path, dict[str, Any]]] = []
    for entry in history:
        for number, path in by_hash.get(entry["sha256"], []):
            mapped.append((number, path, entry))

    if not mapped:
        return [], False

    unique: dict[
        tuple[Path, str, Any, Any],
        tuple[Optional[int], Path, dict[str, Any]],
    ] = {}
    for item in mapped:
        number, path, entry = item
        unique[(
            path,
            entry["sha256"],
            entry["introduced_at"],
            entry["build"],
        )] = item

    return list(unique.values()), True


def _metadata_order(
    candidates: list[tuple[Optional[int], Path, dict[str, Any]]],
    context: Optional[VersionSelectionContext],
) -> list[tuple[Optional[int], Path, dict[str, Any]]]:
    if not candidates:
        return []

    target_date = (
        context.introduced_at
        if context and context.metadata_used
        else None
    )
    target_build = (
        context.build
        if context and context.metadata_used
        else None
    )

    if target_date is not None:
        eligible = []
        for candidate in candidates:
            entry = candidate[2]
            introduced = entry["introduced_at"]
            same_build_without_date = (
                introduced is None
                and target_build is not None
                and entry["build"] == target_build
            )
            if (
                introduced is not None and introduced <= target_date
            ) or same_build_without_date:
                eligible.append(candidate)

        return sorted(
            eligible,
            key=lambda candidate: (
                candidate[2]["introduced_at"]
                or datetime.min.replace(tzinfo=timezone.utc),
                candidate[2]["build"] == target_build
                if target_build is not None else False,
                candidate[0] if candidate[0] is not None else -1,
            ),
            reverse=True,
        )

    if target_build is not None:
        eligible = [
            candidate
            for candidate in candidates
            if candidate[2]["build"] == target_build
        ]
        return sorted(
            eligible,
            key=lambda candidate: (
                candidate[2]["introduced_at"]
                or datetime.min.replace(tzinfo=timezone.utc),
                candidate[0] if candidate[0] is not None else -1,
            ),
            reverse=True,
        )

    dated = [
        candidate for candidate in candidates
        if candidate[2]["introduced_at"] is not None
    ]
    if dated:
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate[2]["introduced_at"]
                or datetime.min.replace(tzinfo=timezone.utc),
                candidate[0] if candidate[0] is not None else -1,
            ),
            reverse=True,
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate[0] if candidate[0] is not None else -1,
        ),
        reverse=True,
    )


def _legacy_candidates(
    logical_path: Path,
    desired_version: Optional[int],
) -> list[tuple[Optional[int], Path]]:
    candidates = _physical_candidates(logical_path)
    if not candidates:
        return []
    if candidates[0][0] is None:
        return candidates

    eligible = [
        (number, path)
        for number, path in candidates
        if desired_version is None
        or (number is not None and number <= desired_version)
    ]
    return sorted(
        eligible,
        key=lambda item: item[0] if item[0] is not None else -1,
        reverse=True,
    )


def dependency_candidates(
    root: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    context: Optional[VersionSelectionContext | int] = None,
) -> list[tuple[Optional[int], Path]]:
    """Return dependency candidates newest-to-oldest in the requested history."""
    root_path = Path(root).resolve()
    normalized = _safe_relative_text(os.fspath(relative_path))
    if normalized is None:
        return []

    logical_path = root_path.joinpath(*PurePosixPath(normalized).parts)
    if logical_path.is_symlink():
        return []

    selection = (
        context
        if isinstance(context, VersionSelectionContext)
        else VersionSelectionContext(
            legacy_version=context,
            search_root=root_path,
        )
    )

    mapped, metadata_valid = _metadata_candidates(
        root_path,
        logical_path,
        selection,
    )
    if metadata_valid:
        ordered = _metadata_order(mapped, selection)
        return [(number, path) for number, path, _ in ordered]

    return _legacy_candidates(logical_path, selection.legacy_version)


def resolve_dependency_path(
    root: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    context: Optional[VersionSelectionContext | int] = None,
) -> Optional[Path]:
    """Resolve one dependency according to metadata, with legacy fallback."""
    candidates = dependency_candidates(root, relative_path, context)
    return candidates[0][1] if candidates else None


class VersionedContentViewCache:
    """Keep the worker's resolver API without creating a temporary archive view."""

    def resolve(
        self,
        input_root: str | os.PathLike[str],
        relative_input_path: str | os.PathLike[str],
    ) -> Tuple[Path, str, Optional[int]]:
        root = Path(input_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Input root is not a directory: {root}")

        text = os.fspath(relative_input_path).replace("\\", "/")
        requested_relative = _safe_relative_text(text)
        if requested_relative is None:
            raise ValueError(f"Unsafe relative asset path: {text}")

        requested_path = root.joinpath(
            *PurePosixPath(requested_relative).parts
        )
        logical_relative, desired_version = split_versioned_relative(
            requested_relative
        )
        logical_path = root.joinpath(
            *PurePosixPath(logical_relative).parts
        )

        if requested_path.is_file() and not requested_path.is_symlink():
            physical_path = requested_path
        elif logical_path.is_file() and not logical_path.is_symlink():
            physical_path = logical_path
        else:
            candidates = dependency_candidates(root, logical_relative, None)
            if not candidates:
                raise FileNotFoundError(
                    f"Asset does not exist: {logical_path}"
                )
            physical_path = candidates[0][1]

        return physical_path.parent, physical_path.name, desired_version

    def cleanup(self) -> None:
        return None


DEFAULT_VERSIONED_CONTENT_VIEWS = VersionedContentViewCache()