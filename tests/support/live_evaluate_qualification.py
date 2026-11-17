"""Observe or pause the real managed worker at live Evaluate boundaries."""

import os
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep

from feedbax.training import live_evaluate
from feedbax.training_host_worker import _Events


def install():
    mode = os.environ["FEEDBAX_QUALIFICATION_MODE"]
    gate = Path(os.environ["FEEDBAX_QUALIFICATION_GATE"])
    phase = _Events.phase
    callback = live_evaluate.live_evaluation_callback
    append = _Events.append
    started = None
    point_completed = {}

    def pause(name):
        reached = gate / f"{name}.reached"
        release = gate / f"{name}.release"
        reached.touch()
        deadline = monotonic() + 15
        while not release.exists():
            if monotonic() >= deadline:
                raise TimeoutError(f"Test did not release {name} gate.")
            sleep(.005)

    def phase_timed(events, name):
        nonlocal started
        if name == "running":
            started = perf_counter_ns()
        if name == "finalizing" and started is not None:
            events.append("qualification_train_window", {"ns": perf_counter_ns() - started})
            started = None
        return phase(events, name)

    def append_timed(events, kind, data, **kwargs):
        append(events, kind, data, **kwargs)
        if kind in {"publication", "evaluation_point"}:
            update = data["checkpoint"]["completed_updates"]
            now = perf_counter_ns()
            point_completed[update] = now
            if kind == "publication":
                append(events, "qualification_checkpoint_clock", {"update": update, "ns": now})

    def append_gated(events, kind, data, **kwargs):
        append(events, kind, data, **kwargs)
        if kind == "publication" and data["key"] == "checkpoint:4":
            pause("stop")

    def instrumented(launch, events):
        if mode == "tamper":
            sidecar = Path(os.environ["FEEDBAX_HOST_LIVE_EVALUATE"])
            sidecar.write_bytes(sidecar.read_bytes() + b" ")
        score = callback(launch, events)
        if score is None:
            return None
        if mode == "lifecycle":
            def gated(checkpoint, model):
                if launch.resume is not None and checkpoint.completed_updates == 6:
                    pause("cancel")
                score(checkpoint, model)
            return gated
        if mode == "timing":
            def timed(checkpoint, model):
                update = checkpoint.completed_updates
                begin = perf_counter_ns()
                score(checkpoint, model)
                end = point_completed[update]
                events.append("qualification_point_window", {"update": update, "ns": end - begin})
                events.append("qualification_checkpoint_clock", {"update": update, "ns": end})
            return timed
        return score

    if mode == "timing":
        _Events.phase = phase_timed
        _Events.append = append_timed
    elif mode == "lifecycle":
        _Events.append = append_gated
    live_evaluate.live_evaluation_callback = instrumented


install()
