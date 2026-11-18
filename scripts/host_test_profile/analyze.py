"""Summarize a host_test_profile record directory by test and by phase.

usage: python scripts/host_test_profile/analyze.py <record-dir> [--json out.json] [--tests ids.txt]
"""

import argparse
import collections
import json
from pathlib import Path

POLLS = {"jobs", "jobs_snapshot", "events", "submissions"}
TERMINAL = {"succeeded", "failed", "stopped", "interrupted", "blocked", "cancelled", "skipped"}


def load(directory: Path) -> dict[int, list[dict]]:
    processes = {}
    for path in directory.glob("*.jsonl"):
        events = []
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        if events:
            processes[events[0]["pid"]] = events
    return processes


def analyze(directory: Path) -> dict:
    processes = load(directory)
    tests = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.defaultdict(collections.Counter)
    popens = {}  # child pid -> popen record
    for events in processes.values():
        for e in events:
            if e["ev"] == "popen":
                if e["role"] == "host" and any(op in e["argv"] for op in ("attach", "init-session")):
                    e["role"] = "host_relay"
                popens[e["child"]] = e
    jax_in_host = []
    jax_host_paths = collections.Counter()
    # Per-process facts.
    for pid, events in processes.items():
        role = "test" if any(e["ev"] == "test_start" for e in events) else popens.get(pid, {}).get("role", "other")
        test = next((e["test"] for e in events if e.get("test")), None)
        for e in events:
            if e["ev"] == "import" and e["module"] == "jax":
                if role == "host":
                    jax_in_host.append(pid)
                    jax_host_paths[tuple(e.get("stack") or ())] += 1
                if test:
                    tests[test][f"jax_import_{'test' if role == 'test' else 'child'}"] += e["t"] - e["start"]
            if e["ev"] == "process_exit" and role == "host" and e.get("jax"):
                jax_in_host.append(pid)
        for e in events:
            if not e.get("test"):
                continue
            t = tests[e["test"]]
            c = counts[e["test"]]
            if e["ev"] == "jax_duration" and e["name"].endswith("backend_compile_duration"):
                t["compile_test" if role == "test" else "compile_worker"] += e["seconds"]
                c["compiles_test" if role == "test" else "compiles_worker"] += 1
            elif e["ev"] == "jax_event" and e["name"].endswith(("cache_hits", "cache_misses")):
                where = "test" if role == "test" else "worker"
                c[f"cache_{'hits' if e['name'].endswith('hits') else 'misses'}_{where}"] += 1
            elif e["ev"] == "popen" and e["role"] == "host":
                c["hosts"] += 1
            elif e["ev"] == "popen" and e["role"] == "host_relay":
                c["host_relays"] += 1
            elif e["ev"] == "host_run" and role == "test":
                c["hosts_in_process"] += 1
            elif e["ev"] == "popen" and e["role"] == "launcher":
                c["workers"] += 1
            elif e["ev"] == "host_leaked":
                counts[e.get("owner") or e["test"]]["hosts_leaked"] += 1
            elif e["ev"] == "reaped":
                c[f"reaped_{e['role']}"] += 1
            elif e["ev"] == "test_phase":
                t[f"pytest_{e['when']}"] += e["stop"] - e["start"]
                if e["outcome"] == "failed":
                    c["failed"] += 1
            elif e["ev"] == "rpc":
                c[f"rpc_{e['op']}"] += 1
                if e["op"] in POLLS:
                    c["polls"] += 1
                    t["poll_rpc_seconds"] += e["t"] - e["start"]
                if e["op"] == "stop":
                    t["stop_rpc"] += e["t"] - e["start"]

    # Host start: Host popen to the end of the RPC that was waiting on it.
    for pid, events in processes.items():
        rpcs = [e for e in events if e["ev"] == "rpc"]
        for e in events:
            if e["ev"] == "popen" and e["role"] == "host" and e.get("test"):
                covering = [r for r in rpcs if r["start"] <= e["start"] <= r["t"]]
                if covering:
                    tests[e["test"]]["host_start"] += covering[0]["t"] - e["start"]
                run = next((h for h in processes.get(e["child"], []) if h["ev"] == "host_run"), None)
                if run is not None:
                    tests[e["test"]]["host_boot"] += run["t"] - e["start"]

    # Workers: Host's launcher popen to the worker's first ready.
    for launcher_pid, launch in popens.items():
        if launch["role"] != "launcher" or not launch.get("test"):
            continue
        test = launch["test"]
        worker_pids = [p for p, e in popens.items() if e["role"] == "worker" and e["pid"] == launcher_pid]
        ready = [e["t"] for p in worker_pids for e in processes.get(p, []) if e["ev"] == "worker_ready"]
        used = any(e["ev"] == "event" and e["kind"] == "accepted" for p in worker_pids for e in processes.get(p, []))
        if ready:
            seconds = min(ready) - launch["start"]
            tests[test]["worker_import"] += seconds
            counts[test]["workers_ready"] += 1
            if not used:
                tests[test]["worker_import_unused"] += seconds
        counts[test]["workers_used"] += used

    # Jobs, from the event-log appends written by workers and launchers.
    jobs = collections.defaultdict(dict)
    job_test = {}
    for pid, events in processes.items():
        for e in events:
            if e["ev"] != "event" or not e.get("test"):
                continue
            path = Path(e["path"])
            key = (path.parent.parent.name, path.parent.name)
            job_test[key] = e["test"]
            name = e["kind"] if e["kind"] != "phase" else f"phase:{e['phase']}"
            jobs[key].setdefault(name, e["t"])
            if e["kind"] == "progress":
                jobs[key]["last_progress"] = e["t"]
    finished = {}
    for events in processes.values():
        for e in events:
            if e["ev"] == "job_finished":
                finished.setdefault(e["job"], e["t"])
    noticed = {}
    for events in processes.values():
        for e in events:
            if e["ev"] == "rpc" and e.get("states"):
                for job, state in e["states"]:
                    if state in TERMINAL and job not in noticed:
                        noticed[job] = e["t"]
    for (job, attempt), j in jobs.items():
        test = job_test[(job, attempt)]
        t, c = tests[test], counts[test]
        c["attempts"] += 1
        accepted = j.get("accepted")
        building, running = j.get("phase:building"), j.get("phase:running")
        end_updates = j.get("phase:finalizing") or j.get("result")
        if accepted is None:
            continue
        if building is not None:
            c["train_attempts"] += 1
            t["train_setup"] += building - accepted
            if running is not None:
                t["build_compile"] += running - building
                if end_updates is not None:
                    t["updates"] += end_updates - running
        elif j.get("terminal") is not None:
            t["other_job_body"] += j["terminal"] - accepted
        if (terminal := j.get("terminal")) is not None and end_updates is not None and building is not None:
            t["finalize"] += terminal - end_updates
        if job in finished and job in noticed and noticed[job] >= finished[job]:
            t["notice"] += noticed[job] - finished[job]
            c["noticed"] += 1
    # Compile cache: which worker cache directories are used by more than one test.
    cache_tests = collections.defaultdict(set)
    for pid, events in processes.items():
        start = events[0]
        if start["ev"] == "process_start" and start.get("jax_cache") and start.get("test"):
            cache_tests[start["jax_cache"]].add(start["test"])
    for test in tests:
        t = tests[test]
        t["wall"] = t["pytest_setup"] + t["pytest_call"] + t["pytest_teardown"]
    return {"tests": {k: dict(v) for k, v in tests.items()}, "counts": {k: dict(v) for k, v in counts.items()},
            "jax_in_host_pids": sorted(set(jax_in_host)), "processes": len(processes),
            "worker_cache_dirs": len(cache_tests),
            "jax_host_import_paths": [[list(k), v] for k, v in jax_host_paths.most_common()],
            "worker_cache_dirs_shared_by_tests": sum(len(v) > 1 for v in cache_tests.values())}


PHASES = [
    "host_start", "host_boot", "worker_import", "worker_import_unused", "train_setup", "build_compile", "updates", "finalize", "other_job_body",
    "notice", "poll_rpc_seconds", "stop_rpc", "pytest_setup", "pytest_teardown", "compile_worker", "compile_test",
    "jax_import_test", "jax_import_child", "wall",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--all", action="store_true", help="include tests that start no Host")
    args = parser.parse_args()
    result = analyze(args.directory)
    selected = {
        test for test, c in result["counts"].items()
        if args.all or c.get("hosts") or c.get("hosts_in_process") or c.get("workers")
        or any(k.startswith("rpc_") for k in c)
    }
    result["selected"] = sorted(selected)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True))
    totals = collections.Counter()
    count_totals = collections.Counter()
    for test in selected:
        totals.update(result["tests"].get(test, {}))
        count_totals.update(result["counts"].get(test, {}))
    print(f"tests starting a Host or worker: {len(selected)}; processes recorded: {result['processes']}")
    print(f"worker JAX cache directories: {result['worker_cache_dirs']}, "
          f"shared by more than one test: {result['worker_cache_dirs_shared_by_tests']}")
    print(f"Host processes that imported JAX: {len(result['jax_in_host_pids'])}")
    for stack, n in result["jax_host_import_paths"]:
        print(f"  {n}x via", " > ".join(stack[-4:]))
    print("summed seconds:", {p: round(totals[p], 1) for p in PHASES})
    print("summed counts:", dict(sorted(count_totals.items())))
    header = ["test", "wall", "hosts", "leaked", "workers", "attempts", *PHASES[:-1]]
    print("\t".join(header))
    for test in sorted(selected, key=lambda k: -result["tests"].get(k, {}).get("wall", 0)):
        t, c = result["tests"].get(test, {}), result["counts"].get(test, {})
        row = [test, f"{t.get('wall', 0):.1f}", str(c.get("hosts", 0) + c.get("hosts_in_process", 0)),
               str(c.get("hosts_leaked", 0)),
               str(c.get("workers", 0)), str(c.get("attempts", 0)), *(f"{t.get(p, 0):.1f}" for p in PHASES[:-1])]
        print("\t".join(row))


if __name__ == "__main__":
    main()
