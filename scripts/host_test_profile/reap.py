"""End every recorded Host, launcher or worker that is still running after a profiled run.

The plugin ends leaked Hosts that the pytest processes started; this sweeps what other processes started
(Hosts started by CLI subprocesses, and workers whose Host was killed). Each is recorded as `reaped`.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

directory = Path(sys.argv[1])
roles = {"host", "launcher", "worker"}
targets = {}
for path in directory.glob("*.jsonl"):
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("ev") == "popen" and event.get("role") in roles:
            targets[event["child"]] = event


def command(pid: int) -> str | None:
    result = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    return result.stdout.strip() or None


live = {}
for pid, event in targets.items():
    # A recorded pid may have been reused; require the same interpreter and Feedbax role.
    text = command(pid)
    if text and event["argv"][0] in text and ("feedbax.host" in text or "_feedbax_host_runtime" in text):
        live[pid] = event
with (directory / f"reaped-{os.getpid()}.jsonl").open("a") as stream:
    for pid, event in live.items():
        stream.write(json.dumps({"t": time.time(), "pid": os.getpid(), "test": event.get("test"), "ev": "reaped",
                                 "child": pid, "role": event["role"]}) + "\n")
for signum in (signal.SIGTERM, signal.SIGKILL):
    for pid in live:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
    if live:
        time.sleep(3)
    live = {pid: event for pid, event in live.items() if command(pid)}
print(f"host_test_profile: ended {len(live)} processes still running after SIGTERM", file=sys.stderr)
