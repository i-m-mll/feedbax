"""Serve a Playwright worker's Studio fixture programs from one interpreter that imports feedbax once.

The runner imports the fixture modules, then writes ``{"ready": true}`` to file descriptor 3. Each
line on standard input is a call ``{"program", "argv"}``, where ``program`` is one of
``{"module": name}``, ``{"path": file}`` or ``{"code": source}``. The runner runs it as
``__main__`` with that argv and answers on descriptor 3 with ``{"code", "stdout", "stderr"}``.

A call is a function of its program and argv. Every call starts from the environment, working
directory and argv the runner started with, which it restores after each call, so nothing a fixture
sets there reaches the next one; a fixture is given its home as an argument. Module imports and
pure caches persist, which is the point.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
import runpy
import sys
import tempfile
import threading
import time
import traceback
from typing import Iterator

PRELOAD = (
    "web.e2e.studio.fixtures.canonical_evaluate",
    "web.e2e.studio.fixtures.canonical_train",
    "web.e2e.studio.fixtures.evaluate_authoring",
    "web.e2e.studio.fixtures.managed_lifecycle",
    "web.e2e.studio.fixtures.managed_light",
    "web.e2e.studio.fixtures.managed_report",
    "web.e2e.studio.fixtures.stored_model",
    "tests.fixtures.scene_geometry",
)


def exit_with_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(1)
    os._exit(1)


@contextmanager
def captured(descriptor: int) -> Iterator[tempfile._TemporaryFileWrapper]:
    """Point a standard descriptor at a file for one call, so output from C code and children is kept too."""
    with tempfile.TemporaryFile() as file:
        saved = os.dup(descriptor)
        os.dup2(file.fileno(), descriptor)
        try:
            yield file
        finally:
            os.dup2(saved, descriptor)
            os.close(saved)


def run(program: dict[str, str]) -> int:
    try:
        if "module" in program:
            runpy.run_module(program["module"], run_name="__main__", alter_sys=True)
        elif "path" in program:
            runpy.run_path(program["path"], run_name="__main__")
        else:
            exec(compile(program["code"], "<fixture>", "exec"), {"__name__": "__main__"})
    except SystemExit as exit:
        if exit.code is None or isinstance(exit.code, int):
            return exit.code or 0
        print(exit.code, file=sys.stderr)
        return 1
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


def call(request: dict) -> dict:
    environment, directory, argv = dict(os.environ), os.getcwd(), sys.argv
    program = request["program"]
    with captured(1) as stdout, captured(2) as stderr:
        sys.argv = [next(iter(program.values())), *request["argv"]]
        try:
            code = run(program)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.argv = argv
            os.chdir(directory)
            os.environ.clear()
            os.environ.update(environment)
        stdout.seek(0)
        stderr.seek(0)
        return {"code": code, "stdout": stdout.read().decode(), "stderr": stderr.read().decode(errors="replace")}


def main() -> None:
    threading.Thread(target=exit_with_parent, args=(os.getppid(),), daemon=True).start()
    replies = os.fdopen(3, "w")
    for module in PRELOAD:
        importlib.import_module(module)
    replies.write(json.dumps({"ready": True}) + "\n")
    replies.flush()
    for line in sys.stdin:
        replies.write(json.dumps(call(json.loads(line))) + "\n")
        replies.flush()


if __name__ == "__main__":
    main()
