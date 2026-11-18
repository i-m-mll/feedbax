"""Summarize a host_test_profile record directory by test and by phase.

usage: python scripts/host_test_profile/analyze.py <record-dir> [--json out.json] [--all] [--slow SECONDS]

Besides the per-test table, it reports the Hosts' own loop: each RPC's time in its handler, each scheduler tick,
and every other callback that held the loop; and it attributes each client RPC slower than --slow to what held
the Host's loop while the client waited.
"""

import argparse
import bisect
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


def union(intervals: list[tuple[float, float]]) -> float:
    """Wall time covered by at least one interval."""
    total, current = 0.0, None
    for start, stop in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, stop]
        else:
            current[1] = max(current[1], stop)
    return total + (current[1] - current[0] if current is not None else 0.0)


def analyze(processes: dict[int, list[dict]]) -> dict:
    tests = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.defaultdict(collections.Counter)
    popens = {}  # child pid -> popen record
    for events in processes.values():
        for e in events:
            if e["ev"] == "popen":
                popens[e["child"]] = e
    jax_in_host = []
    spans = collections.defaultdict(lambda: collections.defaultdict(list))  # test -> phase -> intervals
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
                    spans[e["test"]]["host_start"].append((e["start"], covering[0]["t"]))
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
            spans[test]["worker_import"].append((launch["start"], min(ready)))
            counts[test]["workers_ready"] += 1
            if not used:
                tests[test]["worker_import_unused"] += seconds
        counts[test]["workers_used"] += used

    # Jobs, from the event-log appends written by workers and launchers.
    jobs = collections.defaultdict(dict)
    job_test = {}
    for pid, events in processes.items():
        for e in events:
            if e["ev"] != "event" or not e.get("test") or e.get("won") is False:
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
                spans[test]["build_compile"].append((building, running))
                if end_updates is not None:
                    t["updates"] += end_updates - running
                    spans[test]["updates"].append((running, end_updates))
        elif j.get("terminal") is not None and j["terminal"] >= accepted:
            t["other_job_body"] += j["terminal"] - accepted
            spans[test]["other_job_body"].append((accepted, j["terminal"]))
        if j.get("terminal") is not None and j["terminal"] >= accepted:
            spans[test]["any_job"].append((accepted, j["terminal"]))
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
    # Wall-clock phases are clipped to the test's own window: a job left running past teardown is not its cost.
    windows = collections.defaultdict(list)
    for events in processes.values():
        for e in events:
            if e["ev"] == "test_phase":
                windows[e["nodeid"]] += [e["start"], e["stop"]]
    for test, phases in spans.items():
        if not windows[test]:
            continue
        low, high = min(windows[test]), max(windows[test])

        def clip(intervals):
            return [(max(a, low), min(b, high)) for a, b in intervals if min(b, high) > max(a, low)]

        for phase, intervals in phases.items():
            tests[test][f"{phase}_wall"] = union(clip(intervals))
        tests[test]["host_or_worker_busy_wall"] = union(
            clip(phases["host_start"] + phases["worker_import"] + phases["any_job"]))
    for test in tests:
        t = tests[test]
        t["wall"] = t["pytest_setup"] + t["pytest_call"] + t["pytest_teardown"]
    return {"tests": {k: dict(v) for k, v in tests.items()}, "counts": {k: dict(v) for k, v in counts.items()},
            "jax_in_host_pids": sorted(set(jax_in_host)), "processes": len(processes),
            "worker_cache_dirs": len(cache_tests),
            "jax_host_import_paths": [[list(k), v] for k, v in jax_host_paths.most_common()],
            "worker_cache_dirs_shared_by_tests": sum(len(v) > 1 for v in cache_tests.values())}


PHASES = [
    "host_start_wall", "worker_import_wall", "build_compile_wall", "updates_wall", "other_job_body_wall",
    "any_job_wall", "host_or_worker_busy_wall",
    "host_start", "host_boot", "worker_import", "worker_import_unused", "train_setup", "build_compile", "updates", "finalize", "other_job_body",
    "notice", "poll_rpc_seconds", "stop_rpc", "pytest_setup", "pytest_teardown", "compile_worker", "compile_test",
    "jax_import_test", "jax_import_child", "wall",
]


def _interval(e: dict) -> tuple[float, float]:
    return e["start"], e["start"] + e["seconds"]


def host_loops(processes: dict[int, list[dict]]) -> dict:
    """Per Host process: its root, run interval, handler records, ticks and other loop holds."""
    hosts = {}
    for pid, events in processes.items():
        runs = [e for e in events if e["ev"] == "host_run"]
        if not runs:
            continue
        end = next((e["t"] for e in events if e["ev"] == "host_run_end"), events[-1]["t"])
        ticks = sorted((e for e in events if e["ev"] == "tick"), key=lambda e: e["start"])
        hosts[pid] = {
            "pid": pid, "root": runs[0]["root"], "start": runs[0]["t"], "end": end,
            "rpcs": sorted((e for e in events if e["ev"] == "host_rpc"), key=lambda e: e["start"]),
            "ticks": ticks, "tick_starts": [e["start"] for e in ticks],
            "holds": sorted((e for e in events if e["ev"] == "loop_hold"), key=lambda e: e["start"]),
            "missing": [e for e in events if e["ev"] == "hook_seam_missing"],
        }
    return hosts


def loop_summary(hosts: dict) -> dict:
    ops = collections.defaultdict(lambda: {"count": 0, "held": 0.0, "max_held": 0.0, "wall": 0.0, "max_wall": 0.0})
    ticks = [e["seconds"] for h in hosts.values() for e in h["ticks"]]
    holds = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for h in hosts.values():
        for e in h["rpcs"]:
            o = ops[e["op"]]
            o["count"] += 1
            o["held"] += e["held"]
            o["wall"] += e["seconds"]
            o["max_held"] = max(o["max_held"], e["held"])
            o["max_wall"] = max(o["max_wall"], e["seconds"])
        for e in h["holds"]:
            entry = holds[e["label"]]
            entry[0] += 1
            entry[1] += e["seconds"]
            entry[2] = max(entry[2], e["seconds"])
    ticks.sort()
    return {
        "hosts": len(hosts), "seam_missing": [m for h in hosts.values() for m in h["missing"]],
        "ops": dict(ops),
        "ticks": {"count": len(ticks), "seconds": sum(ticks), "max": ticks[-1] if ticks else 0.0,
                  "p99": ticks[int(0.99 * (len(ticks) - 1))] if ticks else 0.0,
                  "over_1s": sum(t >= 1 for t in ticks), "over_100ms": sum(t >= 0.1 for t in ticks)},
        "holds": {k: {"count": n, "seconds": s, "max": m} for k, (n, s, m) in holds.items()},
    }


def attribute(processes: dict[int, list[dict]], hosts: dict, slow: float) -> list[dict]:
    """Each client RPC slower than `slow` (beyond its own long-poll wait), with what held its Host's loop."""
    by_root = collections.defaultdict(list)
    for h in hosts.values():
        by_root[h["root"]].append(h)
    hosts_started = collections.defaultdict(list)  # root -> Host popen times
    for events in processes.values():
        for e in events:
            if e["ev"] == "popen" and e["role"] == "host" and "--root" in e["argv"]:
                index = e["argv"].index("--root")
                if index + 1 < len(e["argv"]):
                    hosts_started[e["argv"][index + 1]].append(e["start"])
    found = []
    for pid, events in processes.items():
        for e in events:
            if e["ev"] != "rpc":
                continue
            elapsed = e["t"] - e["start"]
            excess = elapsed - (e.get("wait") or 0)
            if excess < slow:
                continue
            low, high = e["start"], e["t"]
            entry = {"test": e.get("test"), "client": pid, "op": e["op"], "seconds": elapsed, "excess": excess,
                     "error": e.get("error"), "root": e["root"], "start": low}
            if e.get("stdio"):
                entry["cause"] = "stdio relay: the Host behind it is not matched"
                found.append(entry)
                continue
            host = next((h for h in by_root.get(e["root"], []) if h["start"] <= high and h["end"] >= low), None)
            if host is None:
                starting = [s for s in hosts_started.get(e["root"], []) if low <= s <= high]
                entry["cause"] = ("Host started during the call and has no loop record" if starting
                                  else "no profiled Host on this root during the call")
                found.append(entry)
                continue
            entry["host"] = host["pid"]
            if host["start"] > low:
                entry["host_boot"] = host["start"] - low
            handler = [r for r in host["rpcs"] if r["op"] == e["op"] and low <= r["start"] <= high]
            if handler:
                r = min(handler, key=lambda r: abs(r["start"] + r["seconds"] - high))
                entry["handler"] = {"received": r["start"] - low, "held": r["held"], "wall": r["seconds"],
                                    "error": r.get("error")}
            holders = []
            first = max(0, bisect.bisect_left(host["tick_starts"], low) - 1)
            for tick in host["ticks"][first:]:
                a, b = _interval(tick)
                if a > high:
                    break
                overlap = min(b, high) - max(a, low)
                if overlap > 0.01:
                    holders.append({"what": "tick", "overlap": overlap, "seconds": tick["seconds"], "at": a - low,
                                    "parts": tick.get("parts") or {}, "error": tick.get("error")})
            for hold in host["holds"]:
                a, b = _interval(hold)
                overlap = min(b, high) - max(a, low)
                if overlap > 0.01:
                    holders.append({"what": hold["label"], "overlap": overlap, "seconds": hold["seconds"],
                                    "at": a - low})
            holders.sort(key=lambda h: -h["overlap"])
            entry["holders"] = holders
            entry["held"] = sum(h["overlap"] for h in holders)
            found.append(entry)
    found.sort(key=lambda f: -f["excess"])
    return found


def describe(entry: dict) -> list[str]:
    lines = [f"{entry['seconds']:6.2f}s {entry['op']}{' ' + entry['error'] if entry['error'] else ''}  "
             f"{entry['test'] or '(no test)'}  client {entry['client']} host {entry.get('host', '-')}"]
    if "cause" in entry:
        return lines + [f"        {entry['cause']}"]
    boot = entry.get("host_boot", 0.0)
    if boot:
        lines.append(f"        Host was starting: its loop began {boot:.2f}s into the call")
    h = entry.get("handler")
    if h:
        lines.append(f"        handler received at +{h['received']:.2f}s, held the loop {h['held']:.3f}s "
                     f"of {h['wall']:.3f}s{' ' + h['error'] if h['error'] else ''}")
    else:
        lines.append("        no handler record: the request never reached the Host's handler")
    if entry["holders"]:
        lines.append(f"        loop held {entry['held']:.2f}s during the call by:")
        for holder in entry["holders"][:6]:
            if holder["what"] == "tick":
                parts = ", ".join(f"{k} {v:.2f}s x{n}" for k, (v, n) in list(holder["parts"].items())[:4])
                lines.append(f"          tick {holder['seconds']:.2f}s at {holder['at']:+.2f}s"
                             f"{' ' + holder['error'] if holder['error'] else ''}{' [' + parts + ']' if parts else ''}")
            else:
                lines.append(f"          {holder['what']} {holder['seconds']:.2f}s at {holder['at']:+.2f}s")
    elif not boot:
        lines.append("        nothing held the Host's loop")
    rest = entry["seconds"] - boot - entry["held"]
    if rest >= 0.1:
        lines.append(f"        {rest:.2f}s outside the Host's loop (connect, transfer, client, machine load)")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--all", action="store_true", help="include tests that start no Host")
    parser.add_argument("--slow", type=float, default=1.0, help="attribute client RPCs slower than this")
    args = parser.parse_args()
    processes = load(args.directory)
    result = analyze(processes)
    hosts = host_loops(processes)
    result["host_loop"] = loop_summary(hosts)
    result["slow_rpcs"] = attribute(processes, hosts, args.slow)
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
    loop = result["host_loop"]
    print(f"Host loops recorded: {loop['hosts']}")
    for missing in loop["seam_missing"]:
        print(f"  HOOK SEAM MISSING in a Host: {missing['cls']}.{', '.join(missing['names'])}")
    ticks = loop["ticks"]
    print(f"  ticks: {ticks['count']}, {ticks['seconds']:.1f}s summed, p99 {ticks['p99']:.3f}s, max {ticks['max']:.2f}s, "
          f">=100ms {ticks['over_100ms']}, >=1s {ticks['over_1s']}")
    print("  handlers (op: count, loop held summed/max, wall summed/max):")
    for op, o in sorted(loop["ops"].items(), key=lambda item: -item[1]["held"]):
        print(f"    {op}: {o['count']}, held {o['held']:.2f}/{o['max_held']:.3f}s, wall {o['wall']:.1f}/{o['max_wall']:.2f}s")
    if loop["holds"]:
        print("  other loop holds >= 20ms (label: count, summed, max):")
        for label, h in sorted(loop["holds"].items(), key=lambda item: -item[1]["seconds"]):
            print(f"    {label}: {h['count']}, {h['seconds']:.2f}s, {h['max']:.2f}s")
    print(f"client RPCs slower than {args.slow}s beyond their own wait: {len(result['slow_rpcs'])}")
    for entry in result["slow_rpcs"]:
        print("\n".join(describe(entry)))
    header = ["test", "wall", "hosts", "relays", "leaked", "workers", "attempts", *PHASES[:-1]]
    print("\t".join(header))
    for test in sorted(selected, key=lambda k: -result["tests"].get(k, {}).get("wall", 0)):
        t, c = result["tests"].get(test, {}), result["counts"].get(test, {})
        row = [test, f"{t.get('wall', 0):.1f}", str(c.get("hosts", 0) + c.get("hosts_in_process", 0)),
               str(c.get("host_relays", 0)), str(c.get("hosts_leaked", 0)),
               str(c.get("workers", 0)), str(c.get("attempts", 0)), *(f"{t.get(p, 0):.1f}" for p in PHASES[:-1])]
        print("\t".join(row))


if __name__ == "__main__":
    main()
