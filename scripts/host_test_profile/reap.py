"""End every recorded Host, launcher or worker that is still running after a profiled run, and fail if any was.

The suite's own sweep ends what runs under the invocation's temporary root; this catches what a profiled run
started elsewhere. Each process ended is recorded as `reaped` and listed, and the exit status is 1.
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
ended = dict(live)
for signum in (signal.SIGTERM, signal.SIGKILL):
    for pid in live:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
    if live:
        time.sleep(3)
    live = {pid: event for pid, event in live.items() if command(pid)}
for pid, event in ended.items():
    print(f"host_test_profile: ended {event['role']} {pid} started by {event.get('test')}: {event['argv']}",
          file=sys.stderr)
if live:
    print(f"host_test_profile: still running after SIGKILL: {sorted(live)}", file=sys.stderr)
sys.exit(1 if ended else 0)
