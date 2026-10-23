from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "feedbax" / "_vendor" / "jax_cookbook"
UPSTREAM_COMMIT = "95060b2448cafb0783ecdc7220818eab9a61c630"

UNMODIFIED_UPSTREAM_SHA256 = {
    "__init__.py": "76b34db865b026007b43edf7c4c2cc5c538aa950ea72c789a15adf5f62664d01",
    "_io.py": "adebaf74d52a2a81206cc7dd7504ad026b0899a967fab81e1ae947abca6d6536",
    "_ldict.py": "5ed2eacda99ad31b945ba5b5545f488e240c7f13813b5d05a6728f57b4e4525d",
    "_print.py": "9bfd4cc26c40d680aabaf8a8ba28aefd593b72e1d684510ac284792d3d217de5",
    "_types.py": "a6afad4381f806b082a3f2704811408ee6064a7604104ea88edf80c4a6be5a7d",
    "_where.py": "62aec41738f4607f2a300712fd3aa6c9516db74ea04298b906874ab482581504",
    "misc.py": "8ea3b262dc83c92d9472206daa9f549abd321c0aefcd59a6a17ab4ade7e18f5f",
    "tree.py": "27b25de7b3273d9162483e5952849e15a9be3a67fb6538be010d6b3a69ff1f30",
}


def test_bundled_cookbook_retains_pinned_source_and_license_provenance() -> None:
    origin = (VENDOR_ROOT / "ORIGIN.md").read_text(encoding="utf-8")
    assert UPSTREAM_COMMIT in origin
    assert (VENDOR_ROOT / "LICENSE").read_text(encoding="utf-8").startswith(
        "                                 Apache License\n"
    )

    for relative_path, expected_sha256 in UNMODIFIED_UPSTREAM_SHA256.items():
        assert hashlib.sha256((VENDOR_ROOT / relative_path).read_bytes()).hexdigest() == (
            expected_sha256
        )


def test_feedbax_has_no_external_cookbook_import_or_distribution_dependency() -> None:
    for path in sorted((REPO_ROOT / "feedbax").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            else:
                continue
            assert all(
                name != "jax_cookbook" and not name.startswith("jax_cookbook.")
                for name in imported
            ), path

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert all(
        not dependency.startswith("jax-cookbook")
        for dependency in pyproject["project"]["dependencies"]
    )
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "jax-cookbook"' not in lock
    assert "github.com/i-m-mll/jax-cookbook" not in lock


def test_external_cookbook_namespace_is_not_installed() -> None:
    assert importlib.util.find_spec("jax_cookbook") is None
