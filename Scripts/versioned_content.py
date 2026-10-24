"""Resolve merged Valve archive versions through a canonical temporary tree."""

from __future__ import annotations

import os
import re
import shutil
import tempfile

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple


_COMPOUND_SUFFIXES = (
    ".dx80.vtx",
    ".dx90.vtx",
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
    """Return N for file.ext/N.ext, or None if it is not a version child."""
    suffix = _version_suffix(container_name)

    if not suffix:
        return None

    if not child_name.casefold().endswith(suffix.casefold()):
        return None

    number_text = child_name[:-len(suffix)]

    if not _VERSION_RE.fullmatch(number_text):
        return None

    return int(number_text)


def split_versioned_relative(
    relative_path: str | os.PathLike[str],
) -> Tuple[str, Optional[int]]:
    """Convert a physical version path to its logical path and version."""
    text = os.fspath(relative_path).replace("\\", "/")
    pure = PurePosixPath(text)

    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative asset path: {text}")

    if len(pure.parts) < 2:
        return pure.as_posix(), None

    container_name = pure.parts[-2]
    child_name = pure.parts[-1]

    version = _variant_number(container_name, child_name)

    if version is None:
        return pure.as_posix(), None

    logical = PurePosixPath(
        *pure.parts[:-2],
        container_name,
    )

    return logical.as_posix(), version


def _find_variants(container: Path) -> Optional[Dict[int, Path]]:
    """Recognize a version container and return {version: physical file}."""
    if not container.is_dir() or container.is_symlink():
        return None

    variants: Dict[int, Path] = {}

    try:
        children = list(container.iterdir())
    except OSError:
        return None

    for child in children:
        if not child.is_file() or child.is_symlink():
            continue

        version = _variant_number(container.name, child.name)

        if version is not None:
            variants[version] = child

    return variants or None


def choose_version(
    versions,
    desired_version: Optional[int],
) -> Optional[int]:
    """Choose latest, or exact/lower for a requested context; never choose higher."""
    available = sorted(set(int(version) for version in versions))

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


def _link_or_copy(source: Path, destination: Path) -> None:
    """Use a hard link when possible; otherwise copy the source file."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@dataclass
class _VersionContainer:
    target: Path
    variants: Dict[int, Path]
    selected_version: Optional[int]


@dataclass
class _ResolvedView:
    root: Path
    desired_version: Optional[int]
    version_containers: list[_VersionContainer]
    temporary_directory: tempfile.TemporaryDirectory

    def activate_version(self, desired_version: Optional[int]) -> None:
        """Switch selected version files without rebuilding the whole view."""
        if self.desired_version == desired_version:
            return

        for container in self.version_containers:
            selected = choose_version(
                container.variants.keys(),
                desired_version,
            )

            if selected == container.selected_version:
                continue

            target = container.target

            try:
                if target.is_file() or target.is_symlink():
                    target.unlink()
                elif target.exists():
                    raise IsADirectoryError(
                        "Version-view target unexpectedly became a directory: "
                        f"{target}"
                    )

                if selected is not None:
                    _link_or_copy(
                        container.variants[selected],
                        target,
                    )

                container.selected_version = selected

            except OSError as exc:
                raise OSError(
                    f"Could not switch versioned asset {target} "
                    f"to version {selected}: {exc}"
                ) from exc

        self.desired_version = desired_version

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()


class VersionedContentViewCache:
    """Maintain a canonical, version-resolved view for each input root."""

    def __init__(self) -> None:
        self._views: Dict[str, _ResolvedView] = {}

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

        key = os.path.normcase(str(root))
        view = self._views.get(key)

        if view is None:
            view = self._build_view(root, desired_version)
            self._views[key] = view
        else:
            view.activate_version(desired_version)

        logical_parts = PurePosixPath(logical_relative).parts
        resolved_path = view.root.joinpath(*logical_parts)

        if not resolved_path.is_file():
            raise FileNotFoundError(
                "The version-resolved asset is missing: "
                f"{logical_relative!r} "
                f"(context version {desired_version!r})."
            )

        return view.root, logical_relative, desired_version

    def cleanup(self) -> None:
        for view in self._views.values():
            view.cleanup()

        self._views.clear()

    def _build_view(
        self,
        input_root: Path,
        desired_version: Optional[int],
    ) -> _ResolvedView:
        try:
            temporary = tempfile.TemporaryDirectory(
                prefix=".vct-version-view-",
                dir=str(input_root.parent),
            )
        except OSError:
            temporary = tempfile.TemporaryDirectory(
                prefix="vct-version-view-"
            )

        destination_root = Path(temporary.name)
        version_containers: list[_VersionContainer] = []

        try:
            self._mirror_directory(
                input_root,
                destination_root,
                desired_version,
                version_containers,
            )
        except Exception:
            temporary.cleanup()
            raise

        return _ResolvedView(
            root=destination_root,
            desired_version=desired_version,
            version_containers=version_containers,
            temporary_directory=temporary,
        )

    def _mirror_directory(
        self,
        source_directory: Path,
        destination_directory: Path,
        desired_version: Optional[int],
        version_containers: list[_VersionContainer],
    ) -> None:
        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            entries = sorted(
                source_directory.iterdir(),
                key=lambda item: item.name.casefold(),
            )
        except OSError as exc:
            raise OSError(
                f"Cannot read archive directory {source_directory}: {exc}"
            ) from exc

        for source in entries:
            if source.is_symlink():
                continue

            if source.is_dir():
                variants = _find_variants(source)

                if variants is not None:
                    selected = choose_version(
                        variants.keys(),
                        desired_version,
                    )

                    target = destination_directory / source.name

                    if selected is not None:
                        _link_or_copy(
                            variants[selected],
                            target,
                        )

                    version_containers.append(
                        _VersionContainer(
                            target=target,
                            variants=variants,
                            selected_version=selected,
                        )
                    )

                    continue

                self._mirror_directory(
                    source,
                    destination_directory / source.name,
                    desired_version,
                    version_containers,
                )

            elif source.is_file():
                _link_or_copy(
                    source,
                    destination_directory / source.name,
                )


DEFAULT_VERSIONED_CONTENT_VIEWS = VersionedContentViewCache()