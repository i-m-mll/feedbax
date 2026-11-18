"""Record Host, worker and job timings in every Python process of a profiled test run.

Loaded through PYTHONPATH only when FEEDBAX_HOST_PROFILE_DIR is set; each process appends
JSON lines to <dir>/<pid>.jsonl. The analysis lives in scripts/host_test_profile/analyze.py.
"""

import os

DIRECTORY = os.environ.get("FEEDBAX_HOST_PROFILE_DIR")

if DIRECTORY:
    import atexit
    import functools
    import importlib.abc
    import json
    import subprocess
    import sys
    import time

    HOOK = os.path.dirname(os.path.abspath(__file__))
    _PATH = os.path.join(DIRECTORY, f"{os.getpid()}.jsonl")

    def record(event: str, **data) -> None:
        line = json.dumps(
            {"t": time.time(), "pid": os.getpid(), "test": os.environ.get("FEEDBAX_HOST_PROFILE_TEST"),
             "ev": event, **data},
            default=str,
        ) + "\n"
        fd = os.open(_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def _role(argv) -> str:
        text = " ".join(map(str, argv))
        if "-m feedbax.host" in text or text.endswith("feedbax.host"):
            return "host_relay" if any(op in argv for op in ("attach", "init-session")) else "host"
        if "_feedbax_host_runtime" in text:
            for role in ("launcher", "worker", "watchdog"):
                if f" {role} " in f" {text} ":
                    return role
            return "host_runtime"
        return "other"

    record("process_start", argv=sys.argv[:4], ppid=os.getppid(), role=_role(sys.argv), python=sys.executable,
           jax_cache=os.environ.get("JAX_COMPILATION_CACHE_DIR"))
    atexit.register(lambda: record("process_exit", jax="jax" in sys.modules, feedbax="feedbax" in sys.modules))

    _popen_init = subprocess.Popen.__init__

    @functools.wraps(_popen_init)
    def _popen(self, args, *rest, **kwargs):
        # Tests may replace PYTHONPATH for their children; keep the hook reachable after theirs.
        env = kwargs.get("env")
        source = os.environ if env is None else env
        path = source.get("PYTHONPATH") or ""
        if HOOK not in path.split(os.pathsep) or source.get("FEEDBAX_HOST_PROFILE_DIR") != DIRECTORY:
            kwargs["env"] = dict(source, FEEDBAX_HOST_PROFILE_DIR=DIRECTORY,
                                 PYTHONPATH=os.pathsep.join(p for p in (path, HOOK) if p))
        start = time.time()
        _popen_init(self, args, *rest, **kwargs)
        argv = list(args) if isinstance(args, (list, tuple)) else [args]
        record("popen", start=start, child=self.pid, role=_role(argv), argv=[str(a)[:80] for a in argv[:6]])

    subprocess.Popen.__init__ = _popen

    def _patch_io(module) -> None:
        append, write_json = module.append, module.write_json

        def wrapped_append(path, kind, data, *args, **kwargs):
            result = append(path, kind, data, *args, **kwargs)
            record("event", path=str(path), kind=kind)
            return result

        def wrapped_write_json(path, value, *args, **kwargs):
            result = write_json(path, value, *args, **kwargs)
            if str(path).endswith("events.jsonl"):
                record("event", path=str(path), kind=value.get("kind"), won=result)
            return result

        module.append, module.write_json = wrapped_append, wrapped_write_json

    def _patch_worker(module) -> None:
        encode = module.encode

        def wrapped(value):
            if value == {"kind": "ready"}:
                record("worker_ready")
            return encode(value)

        module.encode = wrapped

    def _patch_training(module) -> None:
        events = module._Events
        append = events.append

        def wrapped(self, kind, data, *args, **kwargs):
            result = append(self, kind, data, *args, **kwargs)
            detail = {}
            if kind == "phase":
                detail["phase"] = data.get("phase")
            elif kind == "progress":
                detail["updates"] = data.get("completed_updates")
            record("event", path=str(self.path), kind=kind, **detail)
            return result

        events.append = wrapped

    def _patch_client(module) -> None:
        client = module.HostClient
        call = client._call

        def wrapped(self, op, *args, **kwargs):
            start = time.time()
            error = None
            try:
                value = call(self, op, *args, **kwargs)
                return value
            except BaseException as exc:
                error = type(exc).__name__
                value = None
                raise
            finally:
                states = None
                if isinstance(value, list):
                    states = [[v.get("id"), v.get("state")] for v in value if isinstance(v, dict) and "state" in v]
                elif isinstance(value, dict) and isinstance(value.get("jobs"), list):
                    states = [[v.get("id"), v.get("state")] for v in value["jobs"] if isinstance(v, dict)]
                record("rpc", op=op, start=start, root=str(self.root), error=error, states=states)

        client._call = wrapped

    def _patch_store(module) -> None:
        store = module.Store if hasattr(module, "Store") else None
        if store is None or not hasattr(store, "finished"):
            return
        finished = store.finished

        def wrapped(self, row, outcome):
            result = finished(self, row, outcome)
            record("job_finished", job=row["id"], outcome=outcome.get("outcome"))
            return result

        store.finished = wrapped

    def _patch_daemon(module) -> None:
        host = module.Host
        run = host.run

        async def wrapped(self, *args, **kwargs):
            record("host_run", root=str(self.root))
            try:
                return await run(self, *args, **kwargs)
            finally:
                record("host_run_end", root=str(self.root))

        host.run = wrapped

    def _patch_jax(module) -> None:
        from jax import monitoring

        def on_event(name, **_):
            if "compilation_cache" in name:
                record("jax_event", name=name)

        def on_duration(name, seconds, **_):
            if name in ("/jax/core/compile/backend_compile_duration", "/jax/compilation_cache/cache_retrieval_time_sec"):
                record("jax_duration", name=name, seconds=seconds)

        monitoring.register_event_listener(on_event)
        monitoring.register_event_duration_secs_listener(on_duration)

    PATCHES = {
        "feedbax.host.io": _patch_io,
        "_feedbax_host_runtime.io": _patch_io,
        "_feedbax_host_runtime.worker": _patch_worker,
        "feedbax.training_host_worker": _patch_training,
        "feedbax.host.client": _patch_client,
        "feedbax.host.store": _patch_store,
        "feedbax.host.daemon": _patch_daemon,
        "jax": _patch_jax,
    }
    TIMED = {"jax", "feedbax"}

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in PATCHES and fullname not in TIMED:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            exec_module = loader.exec_module

            def timed(module):
                start = time.time()
                exec_module(module)
                if fullname in TIMED:
                    import traceback

                    # The outermost Feedbax frames say which code path pulled the module in.
                    stack = [f"{f.filename.rsplit('/feedbax/', 1)[-1]}:{f.lineno}:{f.name}"
                             for f in traceback.extract_stack()
                             if "/feedbax/" in f.filename and "host_test_profile" not in f.filename]
                    record("import", module=fullname, start=start, stack=stack[:12])
                if fullname in PATCHES:
                    PATCHES[fullname](module)

            # Loaders are often shared; wrap on a per-spec proxy.
            class Proxy(importlib.abc.Loader):
                def create_module(self, spec):
                    return loader.create_module(spec)

                def exec_module(self, module):
                    timed(module)

            spec.loader = Proxy()
            return spec

    sys.meta_path.insert(0, _Finder())
