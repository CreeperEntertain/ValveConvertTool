"""Resolve versioned files in merged Valve archives without crawling the archive."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple


_COMPOUND_SUFFIXES = (
    ".dx70.vtx",
    ".dx80.vtx",
    ".dx90.vtx",
    ".dx11.vtx",
    ".dx12.vtx",
    ".sw.vtx",
)

_VERSION_RE = re.compile(r"^(\d+)$")


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
    """Return N for file.ext/N.ext, or None if this is not a version child."""
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
    """Return the logical path and the version encoded in its final path pair."""
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
    """Inspect one requested version container; never recurse through the archive."""
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


def choose_version(versions, desired_version: Optional[int]) -> Optional[int]:
    """Choose latest, or the exact/highest-lower version for a given context."""
    available = sorted({int(version) for version in versions})

    if not available:
        return None

    if desired_version is None:
        return available[-1]

    eligible = [
        version
        for version in available
        if version <= desired_version
    ]

    return eligible[-1] if eligible else None


class VersionedContentViewCache:
    """Preserve the existing resolver API without creating a temporary archive view."""

    def resolve(
        self,
        input_root: str | os.PathLike[str],
        relative_input_path: str | os.PathLike[str],
    ) -> Tuple[Path, str, Optional[int]]:
        root = Path(input_root).resolve()

        if not root.is_dir():
            raise NotADirectoryError(
                f"Input root is not a directory: {root}"
            )

        logical_relative, desired_version = split_versioned_relative(
            relative_input_path
        )

        logical_path = root.joinpath(
            *PurePosixPath(logical_relative).parts
        )

        if logical_path.is_file():
            physical_path = logical_path
        else:
            variants = _find_variants(logical_path)

            if variants is None:
                raise FileNotFoundError(
                    f"Asset does not exist: {logical_path}"
                )

            selected = choose_version(
                variants.keys(),
                desired_version,
            )

            if selected is None:
                if desired_version is not None:
                    raise FileNotFoundError(
                        f"No version <= {desired_version} exists "
                        f"in {logical_path}"
                    )

                raise FileNotFoundError(
                    f"No usable variants found in {logical_path}"
                )

            physical_path = variants[selected]

        return (
            physical_path.parent,
            physical_path.name,
            desired_version,
        )

    def cleanup(self) -> None:
        return None


DEFAULT_VERSIONED_CONTENT_VIEWS = VersionedContentViewCache()