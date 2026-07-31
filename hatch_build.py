"""Carry verifiable Git identity through Feedbax sdists and wheels."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import ModuleType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.dont_write_bytecode = True


def _load_provenance_codec(root: Path) -> ModuleType:
    path = root / "feedbax" / "_distribution_provenance.py"
    module = ModuleType("_feedbax_distribution_provenance")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


class CustomBuildHook(BuildHookInterface):
    """Map task-private provenance material into either distribution target."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        root = Path(self.root)
        codec = _load_provenance_codec(root)
        provenance = codec.provenance_for_distribution_build(root)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="feedbax-distribution-provenance-"
        )
        artifact = Path(temporary_directory.name) / codec.PROVENANCE_FILENAME
        artifact.write_bytes(provenance)
        force_include = build_data.setdefault("force_include", {})
        if not isinstance(force_include, dict):
            temporary_directory.cleanup()
            raise RuntimeError("Hatch force_include build data has an unsupported type")
        force_include[str(artifact)] = codec.PROVENANCE_PATH
        self._temporary_directory = temporary_directory

    def finalize(
        self, version: str, build_data: dict[str, object], artifact_path: str
    ) -> None:
        self._temporary_directory.cleanup()
