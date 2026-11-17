"""Run one Studio test fixture program in an interpreter that imported its dependencies in advance.

The runner imports the fixture modules, writes ``ready`` to standard error, then reads one JSON
line ``{"program", "argv", "env"}`` from standard input, where ``program`` is one of
``{"module": name}``, ``{"path": file}`` or ``{"code": source}``, and runs it as ``__main__``.
Test workers keep one runner warm so that a fixture call does not wait for these imports.
Importing feedbax creates JAX arrays, so a warm interpreter cannot be shared by forking.
"""

from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
import threading
import time

PRELOAD = (
    "web.e2e.studio.fixtures.canonical_evaluate",
    "web.e2e.studio.fixtures.canonical_train",
    "web.e2e.studio.fixtures.evaluate_authoring",
    "web.e2e.studio.fixtures.managed_lifecycle",
    "web.e2e.studio.fixtures.stored_model",
)


def exit_with_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(1)
    os._exit(1)


def main() -> None:
    threading.Thread(target=exit_with_parent, args=(os.getppid(),), daemon=True).start()
    for module in PRELOAD:
        importlib.import_module(module)
    print("ready", file=sys.stderr, flush=True)
    request = json.loads(sys.stdin.readline())
    os.environ.clear()
    os.environ.update(request["env"])
    program = request["program"]
    sys.argv = [next(iter(program.values())), *request["argv"]]
    if "module" in program:
        runpy.run_module(program["module"], run_name="__main__", alter_sys=True)
    elif "path" in program:
        runpy.run_path(program["path"], run_name="__main__")
    else:
        exec(compile(program["code"], "<fixture>", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
