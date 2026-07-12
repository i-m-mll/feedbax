from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_governed_document_validation_imports_do_not_load_jax() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH"))))
    source = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        for module_name in (
            "feedbax.contracts",
            "feedbax.contracts.migrations",
            "feedbax.contracts.run_matrix",
            "feedbax.contracts.run_composition",
        ):
            importlib.import_module(module_name)

        print(json.dumps({"jax_loaded": "jax" in sys.modules}))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"jax_loaded": False}
