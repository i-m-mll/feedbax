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
        args = [str(a) for a in argv]
        for i, arg in enumerate(args[:-1]):
            if arg == "-m" and args[i + 1] == "feedbax.host":
                rest = args[i + 2:]
                return "host_relay" if "attach" in rest or "init-session" in rest else "host"
        text = " ".join(args)
        if "-m feedbax.host" in text:
            # The Host command is an argument of another program (ssh): this process relays to a Host.
            return "host_relay"
        if "_feedbax_host_runtime" in text:
            for role in ("launcher", "worker", "watchdog"):
                if f" {role} " in f" {text} ":
                    return role
            return "host_runtime"
        return "other"

    record("process_start", argv=sys.argv[:4], ppid=os.getppid(), role=_role(sys.argv), python=sys.executable,
           jax_cache=os.environ.get("JAX_COMPILATION_CACHE_DIR"))
    atexit.register(lambda: record("process_exit", jax="jax" in sys.modules, feedbax="feedbax" in sys.modules))

    STARTED_HOSTS: list = []  # (Popen, test) for each Host this process started; the plugin reaps leaks
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
        role = _role(argv)
        extra = {}
        if role == "host":
            import threading
            import traceback

            STARTED_HOSTS.append((self, os.environ.get("FEEDBAX_HOST_PROFILE_TEST")))
            extra["thread"] = threading.current_thread().name
            extra["stack"] = [f"{f.filename.rsplit('/', 2)[-2:][-1]}:{f.lineno}:{f.name}"
                              for f in traceback.extract_stack()[:-1]
                              if ("/feedbax/" in f.filename or "/tests/" in f.filename)
                              and "host_test_profile" not in f.filename][-8:]
        record("popen", start=start, child=self.pid, role=role, argv=[str(a)[:80] for a in argv[:6]], **extra)

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
        if not hasattr(client, "_exchange"):
            record("hook_seam_missing", cls="HostClient", names=["_exchange"])
            return
        exchange = client._exchange
        relaying = "attach" in sys.argv  # this process is a stdio relay in front of a Host

        def wrapped(self, message, *args, **kwargs):
            start = time.time()
            error = reply = None
            try:
                reply = exchange(self, message, *args, **kwargs)
                return reply
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                try:
                    request = json.loads(message)
                except ValueError:
                    request = {}
                value = reply.get("value") if isinstance(reply, dict) else None
                if isinstance(reply, dict) and isinstance(reply.get("error"), dict):
                    error = reply["error"].get("code")
                states = None
                if isinstance(value, list):
                    states = [[v.get("id"), v.get("state")] for v in value if isinstance(v, dict) and "state" in v]
                elif isinstance(value, dict) and isinstance(value.get("jobs"), list):
                    states = [[v.get("id"), v.get("state")] for v in value["jobs"] if isinstance(v, dict)]
                record("rpc", op=request.get("op"), start=start, root=str(self.root), wait=request.get("wait"),
                       error=error, stdio=getattr(self, "_attachment", None) is not None, relayed=relaying,
                       states=states)

        client._exchange = wrapped

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

    HOLD_SECONDS = 0.02  # a loop callback at least this long is recorded as a hold
    PARTS_SECONDS = 0.05  # a tick at least this long records the time of each Host or Store method it called

    def _time_methods(cls, tick) -> None:
        """Time every plain method of `cls`; within a tick, add each direct callee's time to the tick's parts."""
        import inspect

        for name, function in list(vars(cls).items()):
            if not inspect.isfunction(function) or inspect.iscoroutinefunction(function) or name.startswith("__"):
                continue
            if name == "_tick":
                continue
            label = f"{cls.__name__}.{name}"

            def timed(*args, _function=function, _label=label, **kwargs):
                if tick.parts is None:
                    return _function(*args, **kwargs)
                depth = tick.depth
                tick.depth = depth + 1
                start = time.perf_counter()
                try:
                    return _function(*args, **kwargs)
                finally:
                    tick.depth = depth
                    if depth == 0:
                        seconds, calls = tick.parts.get(_label, (0.0, 0))
                        tick.parts[_label] = (seconds + time.perf_counter() - start, calls + 1)

            setattr(cls, name, functools.wraps(function)(timed))

    import weakref as _weakref

    STEP_START = _weakref.WeakKeyDictionary()  # task -> perf_counter at the start of its running step
    HELD = _weakref.WeakKeyDictionary()  # task -> loop seconds its steps have held
    OPS = _weakref.WeakKeyDictionary()  # task -> the Host operation it is handling

    class _Tick:
        parts: dict | None = None
        depth = 0

    TICK = _Tick()

    def _watch_loop(schedule_tasks) -> None:
        """Record each loop callback that holds the Host's loop, and the loop time each task has held."""
        import asyncio

        handle = asyncio.events.Handle
        if getattr(handle._run, "_host_profile", False):
            return
        run = handle._run

        def wrapped(self):
            task = getattr(self._callback, "__self__", None)
            if not isinstance(task, asyncio.Task):
                task = None
            start = time.perf_counter()
            if task is not None:
                STEP_START[task] = start
            try:
                return run(self)
            finally:
                seconds = time.perf_counter() - start
                if task is not None:
                    HELD[task] = HELD.get(task, 0.0) + seconds
                if seconds >= HOLD_SECONDS and task not in schedule_tasks:
                    if task is not None:
                        label = OPS.get(task) or getattr(task.get_coro(), "__qualname__", None) or task.get_name()
                    else:
                        label = getattr(self._callback, "__qualname__", None) or repr(self._callback)[:80]
                    record("loop_hold", start=time.time() - seconds, seconds=seconds, label=label)

        wrapped._host_profile = True
        handle._run = wrapped

    def _patch_daemon(module) -> None:
        import asyncio

        host = module.Host
        missing = [name for name in ("run", "_tick", "_call") if not hasattr(host, name)]
        if missing:
            record("hook_seam_missing", cls="Host", names=missing)
            return
        run, tick, call = host.run, host._tick, host._call
        schedule_tasks = _weakref.WeakSet()

        async def wrapped_run(self, *args, **kwargs):
            _watch_loop(schedule_tasks)
            record("host_run", root=str(self.root))
            try:
                return await run(self, *args, **kwargs)
            finally:
                record("host_run_end", root=str(self.root))

        def wrapped_tick(self, *args, **kwargs):
            task = asyncio.current_task() if asyncio._get_running_loop() is not None else None
            if task is not None:
                schedule_tasks.add(task)
            TICK.parts, TICK.depth = {}, 0
            wall, start = time.time(), time.perf_counter()
            error = None
            try:
                return tick(self, *args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                seconds = time.perf_counter() - start
                parts, TICK.parts = TICK.parts, None
                detail = {}
                if seconds >= PARTS_SECONDS:
                    detail["parts"] = {k: [round(v, 4), n] for k, (v, n) in
                                       sorted(parts.items(), key=lambda item: -item[1][0]) if v >= 0.001}
                record("tick", start=wall, seconds=seconds, error=error, **detail)

        async def wrapped_call(self, message, *args, **kwargs):
            op = message.get("op") if isinstance(message, dict) else None
            task = asyncio.current_task()
            OPS[task] = f"rpc:{op}"
            wall, start = time.time(), time.perf_counter()
            held0, step0 = HELD.get(task, 0.0), STEP_START.get(task, start)
            error = None
            try:
                return await call(self, message, *args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                end = time.perf_counter()
                held = HELD.get(task, 0.0) - held0 + (end - STEP_START.get(task, start)) - (start - step0)
                record("host_rpc", op=op, start=wall, seconds=end - start, held=held, error=error)

        host.run, host._tick, host._call = wrapped_run, wrapped_tick, wrapped_call
        _time_methods(host, TICK)
        store = sys.modules.get("feedbax.host.store")
        if store is not None and hasattr(store, "Store"):
            _time_methods(store.Store, TICK)

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
