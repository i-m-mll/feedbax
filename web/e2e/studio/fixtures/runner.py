"""Serve Studio test fixture commands from one interpreter that has already imported them.

Each request line is JSON ``{"id", "program", "argv", "env"}``, where ``program`` is one of
``{"module": name}``, ``{"path": file}`` or ``{"code": source}``. The server forks a child that
runs the program as ``__main__`` with that argv and environment, and answers with one JSON line
``{"id", "code", "stdout", "stderr"}`` when the child exits. Children run concurrently.
"""

from __future__ import annotations

import importlib
import json
import os
import runpy
import selectors
import sys
import traceback

PRELOAD = (
    "feedbax.web.app",
    "web.e2e.studio.fixtures.canonical_evaluate",
    "web.e2e.studio.fixtures.canonical_train",
    "web.e2e.studio.fixtures.evaluate_authoring",
    "web.e2e.studio.fixtures.managed_lifecycle",
    "web.e2e.studio.fixtures.stored_model",
    "tests.test_model_document_plan",
)


def run_program(program: dict[str, str]) -> None:
    if "module" in program:
        runpy.run_module(program["module"], run_name="__main__", alter_sys=True)
    elif "path" in program:
        runpy.run_path(program["path"], run_name="__main__")
    else:
        exec(compile(program["code"], "<fixture>", "exec"), {"__name__": "__main__"})


def run_child(program: dict[str, str], argv: list[str], env: dict[str, str]) -> None:
    os.environ.clear()
    os.environ.update(env)
    sys.argv = [next(iter(program.values())), *argv]
    try:
        run_program(program)
    except SystemExit as exit_:
        code = exit_.code
        if code is not None and not isinstance(code, int):
            print(code, file=sys.stderr)
            code = 1
        sys.stdout.flush()
        os._exit(code or 0)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def spawn(request: dict) -> tuple[int, int, int]:
    out_read, out_write = os.pipe()
    err_read, err_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(out_read)
        os.close(err_read)
        os.dup2(out_write, 1)
        os.dup2(err_write, 2)
        os.close(out_write)
        os.close(err_write)
        sys.stdout = os.fdopen(1, "w", buffering=1)
        sys.stderr = os.fdopen(2, "w", buffering=1)
        run_child(request["program"], request["argv"], request["env"])
    os.close(out_write)
    os.close(err_write)
    return pid, out_read, err_read


def main() -> None:
    for module in PRELOAD:
        importlib.import_module(module)
    selector = selectors.DefaultSelector()
    requests = os.fdopen(sys.stdin.fileno(), "rb", buffering=0)
    selector.register(requests, selectors.EVENT_READ, "requests")
    pending = b""
    children: dict[int, dict] = {}
    print(json.dumps({"ready": True}), flush=True)
    while True:
        for key, _ in selector.select():
            if key.data == "requests":
                chunk = requests.read(65536)
                if not chunk:
                    return
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    request = json.loads(line)
                    pid, out, err = spawn(request)
                    child = {"id": request["id"], "pid": pid, "open": 2, "stdout": [], "stderr": []}
                    selector.register(out, selectors.EVENT_READ, (pid, "stdout"))
                    selector.register(err, selectors.EVENT_READ, (pid, "stderr"))
                    children[pid] = child
                continue
            pid, stream = key.data
            child = children[pid]
            data = os.read(key.fd, 65536)
            if data:
                child[stream].append(data)
                continue
            selector.unregister(key.fd)
            os.close(key.fd)
            child["open"] -= 1
            if child["open"] == 0:
                _, status = os.waitpid(pid, 0)
                del children[pid]
                reply = {
                    "id": child["id"],
                    "code": os.waitstatus_to_exitcode(status),
                    "stdout": b"".join(child["stdout"]).decode(),
                    "stderr": b"".join(child["stderr"]).decode(),
                }
                print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
