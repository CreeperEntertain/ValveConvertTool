import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import sys

_here = Path(__file__).resolve()
_candidates = [
    _here.parents[1] / "Scripts",
    _here.parents[1],
    _here.parents[2] / "Scripts",
]
for _candidate in _candidates:
    if (_candidate / "versioned_content.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

import versioned_content as vc


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metadata(path: Path, files: dict) -> None:
    path.write_text(
        json.dumps({"schemaVersion": 1, "files": files}, indent=2),
        encoding="utf-8",
    )


class MetadataSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        vc._METADATA_CACHE.clear()
        vc._HASH_CACHE.clear()
        vc._WARNED_METADATA.clear()

    def tearDown(self):
        self.temp.cleanup()
        vc._METADATA_CACHE.clear()
        vc._HASH_CACHE.clear()
        vc._WARNED_METADATA.clear()

    def test_selects_by_introduction_date_not_variant_number(self):
        mdl_container = self.root / "models" / "robot.mdl"
        mdl_container.mkdir(parents=True)
        source = mdl_container / "0.mdl"
        source.write_bytes(b"robot mdl, build 200")

        vmt_container = self.root / "materials" / "robot.vmt"
        vmt_container.mkdir(parents=True)
        newer = vmt_container / "0.vmt"
        older = vmt_container / "9.vmt"
        newer.write_bytes(b"material introduced in 2024")
        older.write_bytes(b"material introduced in 2021")

        write_metadata(self.root / ".valve-diff-metadata.json", {
            "models/robot.mdl": [{
                "version": None,
                "sha256": sha(source),
                "introducedAt": "2022-01-01",
                "build": "portal-build-200",
            }],
            "materials/robot.vmt": [
                {
                    "version": 7,
                    "sha256": sha(newer),
                    "introducedAt": "2024-04-01",
                    "build": "portal-build-400",
                },
                {
                    "version": None,
                    "sha256": sha(older),
                    "introducedAt": "2021-05-01",
                    "build": "portal-build-100",
                },
            ],
        })

        context = vc.get_selection_context(self.root, source, 0)
        selected = vc.resolve_dependency_path(
            self.root,
            "materials/robot.vmt",
            context,
        )
        self.assertEqual(selected, older)

    def test_exact_build_is_treated_as_opaque_identifier(self):
        source_container = self.root / "maps" / "arena.bsp"
        source_container.mkdir(parents=True)
        source = source_container / "1.bsp"
        source.write_bytes(b"map")

        dep_container = self.root / "materials" / "arena.vmt"
        dep_container.mkdir(parents=True)
        other_build = dep_container / "0.vmt"
        matching_build = dep_container / "1.vmt"
        other_build.write_bytes(b"not this build")
        matching_build.write_bytes(b"exact build")

        write_metadata(self.root / ".valve-diff-metadata.json", {
            "maps/arena.bsp": [{
                "version": None,
                "sha256": sha(source),
                "introducedAt": None,
                "build": "branch-2024.11-hotfix",
            }],
            "materials/arena.vmt": [
                {
                    "version": 500,
                    "sha256": sha(other_build),
                    "introducedAt": None,
                    "build": "branch-2024.10",
                },
                {
                    "version": None,
                    "sha256": sha(matching_build),
                    "introducedAt": None,
                    "build": "branch-2024.11-hotfix",
                },
            ],
        })

        context = vc.get_selection_context(self.root, source, 1)
        selected = vc.resolve_dependency_path(
            self.root,
            "materials/arena.vmt",
            context,
        )
        self.assertEqual(selected, matching_build)

    def test_no_future_variant_is_selected_for_historical_context(self):
        target_dir = self.root / "maps" / "old.bsp"
        target_dir.mkdir(parents=True)
        target = target_dir / "3.bsp"
        target.write_bytes(b"old target")

        dep_dir = self.root / "materials" / "future.vmt"
        dep_dir.mkdir(parents=True)
        future = dep_dir / "0.vmt"
        future.write_bytes(b"introduced later")

        write_metadata(self.root / ".valve-diff-metadata.json", {
            "maps/old.bsp": [{
                "version": None,
                "sha256": sha(target),
                "introducedAt": "2020-01-01",
                "build": "old",
            }],
            "materials/future.vmt": [{
                "version": None,
                "sha256": sha(future),
                "introducedAt": "2022-01-01",
                "build": "new",
            }],
        })

        context = vc.get_selection_context(self.root, target, 3)
        self.assertIsNone(
            vc.resolve_dependency_path(
                self.root,
                "materials/future.vmt",
                context,
            )
        )

    def test_folder_local_metadata_is_supported(self):
        source_container = self.root / "models" / "bot.mdl"
        source_container.mkdir(parents=True)
        source = source_container / "4.mdl"
        source.write_bytes(b"target in build b")

        write_metadata(source_container / ".valve-diff-metadata.json", {
            "bot.mdl": [{
                "version": None,
                "sha256": sha(source),
                "introducedAt": "2023-06-01",
                "build": "b",
            }],
        })

        dep_container = self.root / "materials" / "bot.vmt"
        dep_container.mkdir(parents=True)
        old = dep_container / "0.vmt"
        new = dep_container / "1.vmt"
        old.write_bytes(b"old material")
        new.write_bytes(b"new material")

        write_metadata(dep_container / ".valve-diff-metadata.json", {
            "bot.vmt": [
                {
                    "version": None,
                    "sha256": sha(old),
                    "introducedAt": "2022-01-01",
                    "build": "a",
                },
                {
                    "version": None,
                    "sha256": sha(new),
                    "introducedAt": "2024-01-01",
                    "build": "c",
                },
            ],
        })

        context = vc.get_selection_context(self.root, source, 4)
        self.assertEqual(
            vc.resolve_dependency_path(
                self.root,
                "materials/bot.vmt",
                context,
            ),
            old,
        )

    def test_stale_hash_falls_back_to_legacy_numbered_selection(self):
        container = self.root / "materials" / "stale.vmt"
        container.mkdir(parents=True)
        first = container / "0.vmt"
        latest = container / "2.vmt"
        first.write_bytes(b"first")
        latest.write_bytes(b"latest")

        write_metadata(self.root / ".valve-diff-metadata.json", {
            "materials/stale.vmt": [{
                "version": 0,
                "sha256": "0" * 64,
                "introducedAt": "2020-01-01",
                "build": "old",
            }],
        })

        self.assertEqual(
            vc.resolve_dependency_path(
                self.root,
                "materials/stale.vmt",
                None,
            ),
            latest,
        )
        self.assertEqual(
            vc.resolve_dependency_path(
                self.root,
                "materials/stale.vmt",
                1,
            ),
            first,
        )

    def test_numbered_physical_input_is_preserved_by_view_resolver(self):
        container = self.root / "maps" / "arena.bsp"
        container.mkdir(parents=True)
        physical = container / "4.bsp"
        physical.write_bytes(b"specific scanned input")

        returned_root, returned_name, legacy_version = (
            vc.DEFAULT_VERSIONED_CONTENT_VIEWS.resolve(
                self.root,
                "maps/arena.bsp/4.bsp",
            )
        )

        self.assertEqual(returned_root / returned_name, physical)
        self.assertEqual(legacy_version, 4)


if __name__ == "__main__":
    unittest.main()