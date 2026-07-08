#!/usr/bin/env bash
# Integration tests that exercise runpod_deploy.sh acquisition + launch paths
# WITHOUT creating a real pod, by putting stub `runpodctl` / `ssh` / `rsync`
# binaries on PATH. These prove:
#   - W1: the acquire_trap tears down a pod we created when acquisition fails
#         (simulated dead-state), and the DC-iteration advances.
#   - W2: a 3-row manifest with cap=2 + stagger launches all rows with per-row
#         sentinels in a local mock (the stub `ssh` executes the remote command
#         against a fake remote root on the local filesystem).
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEPLOY_DIR="$(cd "$TESTS_DIR/.." && pwd -P)"
FIX="$TESTS_DIR/fixtures"

PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
no() { FAIL=$((FAIL + 1)); printf '  FAIL %s\n    %s\n' "$1" "$2"; }
section() { printf '\n== %s ==\n' "$1"; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
section "argument forwarding: no CLI args reaches guarded acquisition"

NOARGS="$WORK/noargs"; mkdir -p "$NOARGS/stubs"
CLOG0="$NOARGS/create.log"; : > "$CLOG0"
cat > "$NOARGS/stubs/runpodctl" <<STUB
#!/usr/bin/env bash
case "\$1 \$2" in
  "pod create") echo "create" >> "$CLOG0"; echo '{"id":"should-not-happen"}' ;;
  "user --output"|"user -o"|"user ") echo 'transport error' >&2; exit 7 ;;
  "datacenter list") cat "$FIX/datacenter_list.json" ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$NOARGS/stubs/runpodctl"
for tool in ssh rsync curl; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$NOARGS/stubs/$tool"; chmod +x "$NOARGS/stubs/$tool"
done

set +e
PATH="$NOARGS/stubs:$PATH" \
RUNPOD_NAME="itest-noargs-$$" \
ACQUIRE_LOCK_FILE="$NOARGS/lock" \
bash "$DEPLOY_DIR/runpod_deploy.sh" \
  >"$NOARGS/out" 2>"$NOARGS/err"
rc=$?
set -e
if [ "$rc" -ne 0 ] && grep -q 'balance could not be read' "$NOARGS/err"; then
  ok "no-args invocation reaches balance guard"
else
  no "no-args invocation should reach balance guard" "rc=$rc; $(tail -5 "$NOARGS/err" | tr '\n' '|')"
fi
if ! grep -q 'original_args' "$NOARGS/err"; then
  ok "no-args invocation does not trip original_args nounset"
else
  no "no-args invocation must not trip original_args nounset" "$(cat "$NOARGS/err" | tr '\n' '|')"
fi
n_create0=$( { grep -c '^create' "$CLOG0" || true; } | head -1); n_create0=${n_create0:-0}
if [ "$n_create0" -eq 0 ]; then
  ok "no-args regression creates no pod"
else
  no "no-args regression must not create a pod" "creates=$n_create0"
fi

# ---------------------------------------------------------------------------
section "W1 trap teardown on simulated dead-state acquisition"

STUBS="$WORK/stubs"
mkdir -p "$STUBS"
TEARDOWN_LOG="$WORK/teardown.log"
CREATE_LOG="$WORK/create.log"
: > "$TEARDOWN_LOG"; : > "$CREATE_LOG"

# Stub runpodctl: `pod create` returns a pod id and records the DC; `pod get`
# returns the EXITED (dead) fixture so every attempt dead-fails; `pod remove`
# records teardown; `user`/`datacenter list` return canned JSON.
cat > "$STUBS/runpodctl" <<STUB
#!/usr/bin/env bash
case "\$1 \$2" in
  "pod create")
    dc="?"; for a in "\$@"; do [ "\$prev" = "--data-center-ids" ] && dc="\$a"; prev="\$a"; done
    echo "create dc=\$dc" >> "$CREATE_LOG"
    echo '{"id":"podstub-'"\$dc"'"}'
    ;;
  "pod get")
    cat "$FIX/pod_exited.json"
    ;;
  "pod remove")
    echo "remove \$3" >> "$TEARDOWN_LOG"
    ;;
  "pod stop")
    echo "stop \$3" >> "$TEARDOWN_LOG"
    ;;
  "user --output"|"user -o"|"user ")
    echo '{"clientBalance":50.0}'
    ;;
  "datacenter list")
    cat "$FIX/datacenter_list.json"
    ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$STUBS/runpodctl"

# Stub ssh / rsync / curl / jq passthrough (jq is real). ssh always succeeds.
for tool in ssh rsync; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$STUBS/$tool"
  chmod +x "$STUBS/$tool"
done
printf '#!/usr/bin/env bash\nexit 0\n' > "$STUBS/curl"
chmod +x "$STUBS/curl"

# Run acquisition. With dead-fixture pod_get, every attempt dead-fails fast and
# the loop exhausts candidate DCs, then dies. The trap must have torn down each
# created pod. Keep the grace short so the test is quick (dead-state is caught
# immediately anyway).
set +e
PATH="$STUBS:$PATH" \
ENDPOINT_GRACE_SECONDS=4 DEAD_STATE_POLL_SECONDS=1 \
ACQUIRE_WALL_CLOCK_CAP_SECONDS=60 \
RUNPOD_GPU_ID="NVIDIA GeForce RTX 5090" \
RUNPOD_NAME="itest-$$" \
ACQUIRE_LOCK_FILE="$WORK/lock" \
SKIP_IMAGE_CHECK=1 \
bash "$DEPLOY_DIR/runpod_deploy.sh" --acquire-only \
  >"$WORK/acq.out" 2>"$WORK/acq.err"
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  ok "acquisition fails (rc=$rc) when every pod dead-states"
else
  no "acquisition should fail on all-dead" "rc=$rc"
fi

# Created 3 pods (EU-RO-1, US-KS-2, AP-IN-1 ranked by 5090 stock).
n_created=$(wc -l < "$CREATE_LOG" | tr -d ' ')
if [ "$n_created" -eq 3 ]; then
  ok "iterated all 3 ranked datacenters ($n_created creates)"
else
  no "should create one pod per ranked DC" "created=$n_created; $(cat "$CREATE_LOG" | tr '\n' '|')"
fi

# Every created pod was torn down by the trap / loop.
n_torn=$(grep -c '^remove ' "$TEARDOWN_LOG" 2>/dev/null || echo 0)
if [ "$n_torn" -ge "$n_created" ] && [ "$n_torn" -gt 0 ]; then
  ok "trap/loop tore down all created pods ($n_torn teardowns)"
else
  no "every created pod must be torn down" "torn=$n_torn created=$n_created; $(cat "$TEARDOWN_LOG" | tr '\n' '|')"
fi

# dead_exited status was emitted.
if grep -q 'acquire_status=dead_exited' "$WORK/acq.err"; then
  ok "emitted dead_exited status vocabulary"
else
  no "should emit dead_exited" "$(grep acquire_status "$WORK/acq.err" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "W2 3-row manifest, cap=2, stagger -> per-row sentinels (local mock)"

# Fake remote: ssh stub executes the remote command locally so per-row sentinels
# land on the real filesystem. Each row command is a quick sleep that touches a
# marker; we verify all three .pid/.done files appear.
REMOTE_FS="$WORK/remote_fs"
SDIR2="$REMOTE_FS/feedbax_runs/runpod-deploy/sentinels"
# The mock remote workdir must exist so each row's `cd <workdir>` succeeds.
mkdir -p "$REMOTE_FS/rlrmp"

STUBS2="$WORK/stubs2"
mkdir -p "$STUBS2"
# ssh stub: strip the leading ssh args and exec the final command argument as a
# local bash command, so nohup/touch run against $REMOTE_FS paths.
cat > "$STUBS2/ssh" <<'STUB'
#!/usr/bin/env bash
# The remote command is the last argument.
cmd="${!#}"
bash -c "$cmd"
STUB
chmod +x "$STUBS2/ssh"
for tool in rsync curl runpodctl; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$STUBS2/$tool"
  chmod +x "$STUBS2/$tool"
done

cat > "$WORK/rows.json" <<JSON
{"schema_version":1,"rows":[
  {"id":"row_a","command":"sleep 0.2"},
  {"id":"row_b","command":"sleep 0.2"},
  {"id":"row_c","command":"sleep 0.2"}]}
JSON
cat > "$WORK/spec.json" <<JSON
{"user_confirmed":true}
JSON

# Drive launch_training directly via a tiny harness that sources the run-prep
# library and supplies the few driver transport helpers it needs.
HARNESS="$WORK/launch_harness.sh"
{
  echo 'set -uo pipefail'
  echo "source '$DEPLOY_DIR/lib_acquire.sh'"
  echo "source '$DEPLOY_DIR/lib_run_prep.sh'"
  # Minimal helpers used by launch_row / launch_training.
  cat <<'H'
DRY_RUN=0
RUNPOD_SSH_KEY="/dev/null"
SSH_CONNECT_TIMEOUT=10
log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
print_cmd() { :; }
run_cmd() { "$@"; }
rsync_rsh() { printf 'ssh\n'; }
sq() { local v=${1-}; v=${v//\'/\'\\\'\'}; printf "'%s'" "$v"; }
expand_path() { printf '%s\n' "$1"; }
remote_cmd() {
  local command=$1
  ssh -i /dev/null -p 22 root@localhost "$command"
}
remote_capture() {
  local command=$1
  ssh -i /dev/null -p 22 root@localhost "$command"
}
H
  echo 'launch_training'
} > "$HARNESS"

set +e
PATH="$STUBS2:$PATH" \
REMOTE_RLRMP_ROOT="$REMOTE_FS/rlrmp" \
REMOTE_RUN_DIR="$REMOTE_FS/feedbax_runs/runpod-deploy" \
REMOTE_SENTINEL_DIR="$SDIR2" \
JAX_COMPILATION_CACHE_DIR="$REMOTE_FS/jax_cache" \
ROWS_MANIFEST="$WORK/rows.json" \
ROW_LAUNCH_STAGGER_SECONDS=0 \
MAX_PARALLEL_ROWS=2 \
SENTINEL_POLL_SECONDS=1 \
SSH_HOST=localhost SSH_PORT=22 \
bash "$HARNESS" >"$WORK/launch.out" 2>"$WORK/launch.err"
lrc=$?
set -e

# Give the backgrounded sleeps time to finish and touch .done.
sleep 1

n_pid=$(ls "$SDIR2"/row_*.pid 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_pid" -eq 3 ]; then
  ok "all 3 rows wrote a .pid (per-row launch)"
else
  no "expected 3 row .pid files" "got $n_pid; $(ls "$SDIR2" 2>/dev/null | tr '\n' ' ')"
fi

n_done=$(ls "$SDIR2"/row_*.done 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_done" -eq 3 ]; then
  ok "all 3 rows reached .done (per-row sentinels)"
else
  no "expected 3 row .done files" "got $n_done; $(ls "$SDIR2" 2>/dev/null | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
section "9895c08 warm-first waits for log readiness, not row pid"

WARM="$WORK/warm_ready"; mkdir -p "$WARM/stubs" "$WARM/remote/rlrmp" "$WARM/guard"
WARM_SENT="$WARM/remote/run/sentinels"
cat > "$WARM/stubs/ssh" <<'STUB'
#!/usr/bin/env bash
cmd="${!#}"
bash -c "$cmd"
STUB
chmod +x "$WARM/stubs/ssh"
for tool in rsync curl runpodctl; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$WARM/stubs/$tool"
  chmod +x "$WARM/stubs/$tool"
done

jq -n \
  --arg cmd_a "sleep 0.5; if [ -f '$WARM/guard/row_b_started' ]; then touch '$WARM/guard/fanout_before_ready'; fi; echo WARM_READY; sleep 0.1" \
  --arg cmd_b "touch '$WARM/guard/row_b_started'; sleep 0.1" \
  '{schema_version: 1, rows: [{id: "row_a", command: $cmd_a}, {id: "row_b", command: $cmd_b}]}' \
  > "$WARM/rows.json"

WARM_HARNESS="$WARM/harness.sh"
{
  echo 'set -uo pipefail'
  echo "source '$DEPLOY_DIR/lib_acquire.sh'"
  echo "source '$DEPLOY_DIR/lib_run_prep.sh'"
  cat <<'H'
DRY_RUN=0
RUNPOD_SSH_KEY="/dev/null"
SSH_CONNECT_TIMEOUT=10
log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
print_cmd() { :; }
run_cmd() { "$@"; }
rsync_rsh() { printf 'ssh\n'; }
sq() { local v=${1-}; v=${v//\'/\'\\\'\'}; printf "'%s'" "$v"; }
expand_path() { printf '%s\n' "$1"; }
remote_cmd() { ssh -i /dev/null -p 22 root@localhost "$1"; }
remote_capture() { ssh -i /dev/null -p 22 root@localhost "$1"; }
H
  echo 'launch_training'
} > "$WARM_HARNESS"

set +e
PATH="$WARM/stubs:$PATH" \
REMOTE_RLRMP_ROOT="$WARM/remote/rlrmp" \
REMOTE_RUN_DIR="$WARM/remote/run" \
REMOTE_SENTINEL_DIR="$WARM_SENT" \
JAX_COMPILATION_CACHE_DIR="$WARM/remote/jax_cache" \
ROWS_MANIFEST="$WARM/rows.json" \
ROW_LAUNCH_STAGGER_SECONDS=0 \
MAX_PARALLEL_ROWS=2 \
SENTINEL_POLL_SECONDS=1 \
WARM_COMPILE_READY_REGEX="WARM_READY" \
SSH_HOST=localhost SSH_PORT=22 \
bash "$WARM_HARNESS" >"$WARM/out" 2>"$WARM/err"
wrc=$?
set -e
sleep 1

if [ "$wrc" -eq 0 ] && [ -f "$WARM/guard/row_b_started" ]; then
  ok "9895c08: warm-ready harness launches row_b after readiness"
else
  no "9895c08: warm-ready harness should launch both rows" "rc=$wrc; $(cat "$WARM/err" | tr '\n' '|')"
fi
if [ ! -f "$WARM/guard/fanout_before_ready" ]; then
  ok "9895c08: row_b did not fan out before row_a log readiness"
else
  no "9895c08: fan-out must not be released by pid alone" "$(cat "$WARM/err" | tr '\n' '|')"
fi
if grep -q 'warm compile row row_a reached active training' "$WARM/err" &&
   grep -q 'WARM_READY' "$WARM/remote/run/logs/row_a.log"; then
  ok "9895c08: readiness came from row log marker"
else
  no "9895c08: expected log readiness evidence" "$(cat "$WARM/err" "$WARM/remote/run/logs/row_a.log" 2>/dev/null | tr '\n' '|')"
fi

section "nonzero launch commands create failed sentinels only"

F_SENT="$WORK/fail_sentinel"; mkdir -p "$F_SENT/stubs"
F_REMOTE="$F_SENT/remote_fs"
F_SDIR="$F_REMOTE/feedbax_runs/runpod-deploy/sentinels"
mkdir -p "$F_REMOTE/rlrmp"
cat > "$F_SENT/stubs/ssh" <<'STUB'
#!/usr/bin/env bash
cmd="${!#}"
bash -c "$cmd"
STUB
chmod +x "$F_SENT/stubs/ssh"
for tool in rsync curl runpodctl; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$F_SENT/stubs/$tool"
  chmod +x "$F_SENT/stubs/$tool"
done

F_HARNESS="$F_SENT/fail_harness.sh"
{
  echo 'set -uo pipefail'
  echo "source '$DEPLOY_DIR/lib_acquire.sh'"
  echo "source '$DEPLOY_DIR/lib_run_prep.sh'"
  cat <<'H'
DRY_RUN=0
RUNPOD_SSH_KEY="/dev/null"
SSH_CONNECT_TIMEOUT=10
log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
sq() { local v=${1-}; v=${v//\'/\'\\\'\'}; printf "'%s'" "$v"; }
remote_cmd() { ssh -i /dev/null -p 22 root@localhost "$1"; }
remote_capture() { ssh -i /dev/null -p 22 root@localhost "$1"; }
H
  cat <<'H'
remote_nohup_sentinel "failing bootstrap" "$REMOTE_RLRMP_ROOT" "false" \
  "$REMOTE_SENTINEL_DIR/bootstrap.done" "$REMOTE_SENTINEL_DIR/bootstrap.failed" \
  "$REMOTE_RUN_DIR/logs/bootstrap.log"
launch_row "row_fail" "$REMOTE_RLRMP_ROOT" "false"
H
} > "$F_HARNESS"

set +e
PATH="$F_SENT/stubs:$PATH" \
REMOTE_RLRMP_ROOT="$F_REMOTE/rlrmp" \
REMOTE_RUN_DIR="$F_REMOTE/feedbax_runs/runpod-deploy" \
REMOTE_SENTINEL_DIR="$F_SDIR" \
JAX_COMPILATION_CACHE_DIR="$F_REMOTE/jax_cache" \
SSH_HOST=localhost SSH_PORT=22 \
bash "$F_HARNESS" >"$F_SENT/out" 2>"$F_SENT/err"
frc=$?
set -e
sleep 1

if [ "$frc" -eq 0 ]; then
  ok "failing commands launch asynchronously"
else
  no "failing command harness should launch asynchronously" "rc=$frc; $(cat "$F_SENT/err" | tr '\n' '|')"
fi
if [ -f "$F_SDIR/bootstrap.failed" ] && [ ! -f "$F_SDIR/bootstrap.done" ]; then
  ok "bootstrap failure writes .failed without .done"
else
  no "bootstrap failure sentinel" "$(ls "$F_SDIR" 2>/dev/null | tr '\n' ' ')"
fi
if [ -f "$F_SDIR/row_fail.failed" ] && [ ! -f "$F_SDIR/row_fail.done" ]; then
  ok "row failure writes .failed without .done"
else
  no "row failure sentinel" "$(ls "$F_SDIR" 2>/dev/null | tr '\n' ' ')"
fi
if [ -f "$F_REMOTE/feedbax_runs/runpod-deploy/logs/bootstrap.log" ] &&
   [ -f "$F_REMOTE/feedbax_runs/runpod-deploy/logs/row_fail.log" ]; then
  ok "failure logs are preserved"
else
  no "failure logs should be present" "$(find "$F_REMOTE/feedbax_runs/runpod-deploy" -maxdepth 2 -type f 2>/dev/null | tr '\n' ' ')"
fi

if grep -q 'launched 3 row(s)' "$WORK/launch.err"; then
  ok "launch_training reported 3 rows launched"
else
  no "launch_training should report 3 rows" "$(tail -3 "$WORK/launch.err" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "issue 63ae138 reused-pod probe failure clears and rebuilds venv"

VENV_REMOTE="$WORK/venv_remote"
VENV_SENTINELS="$VENV_REMOTE/feedbax_runs/runpod-deploy/sentinels"
mkdir -p "$VENV_REMOTE/rlrmp" "$VENV_SENTINELS"
STUBS_VENV="$WORK/stubs_venv"
mkdir -p "$STUBS_VENV"
UV_LOG="$WORK/uv.log"; : > "$UV_LOG"

cat > "$STUBS_VENV/ssh" <<'STUB'
#!/usr/bin/env bash
cmd="${!#}"
/bin/bash -c "$cmd"
STUB
chmod +x "$STUBS_VENV/ssh"

cat > "$STUBS_VENV/bash" <<'STUB'
#!/bin/bash
if [ "${1:-}" = "-lc" ]; then
  shift
  exec /bin/bash -c "$@"
fi
exec /bin/bash "$@"
STUB
chmod +x "$STUBS_VENV/bash"

cat > "$STUBS_VENV/uv" <<STUB
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$UV_LOG"
if [ "\$1" = run ] && [ "\$2" = --no-sync ]; then
  exit 9
fi
exit 0
STUB
chmod +x "$STUBS_VENV/uv"

VENV_HARNESS="$WORK/venv_harness.sh"
{
  echo 'set -euo pipefail'
  echo "source '$DEPLOY_DIR/lib_run_prep.sh'"
  cat <<'H'
DRY_RUN=0
RUNPOD_SSH_KEY="/dev/null"
SSH_CONNECT_TIMEOUT=10
SENTINEL_TIMEOUT_SECONDS=10
SENTINEL_POLL_SECONDS=1
POD_CREATED_BY_US=0
log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
print_cmd() { :; }
run_cmd() { "$@"; }
capture_cmd() { "$@"; }
sq() { local v=${1-}; v=${v//\'/\'\\\'\'}; printf "'%s'" "$v"; }
expand_path() { printf '%s\n' "$1"; }
remote_cmd() {
  local command=$1
  ssh -i /dev/null -p 22 root@localhost "$command"
}
remote_capture() {
  local command=$1
  ssh -i /dev/null -p 22 root@localhost "$command"
}
H
  for fn in bootstrap_remote_env mark_bootstrap_branch clear_venv_probe_markers \
            probe_reused_remote_env bootstrap_remote_env_for_pod; do
    sed -n "/^$fn() {/,/^}/p" "$DEPLOY_DIR/runpod_deploy.sh"
  done
  echo 'bootstrap_remote_env_for_pod'
} > "$VENV_HARNESS"

set +e
PATH="$STUBS_VENV:$PATH" \
REMOTE_RLRMP_ROOT="$VENV_REMOTE/rlrmp" \
REMOTE_RUN_DIR="$VENV_REMOTE/feedbax_runs/runpod-deploy" \
REMOTE_SENTINEL_DIR="$VENV_SENTINELS" \
SSH_HOST=localhost SSH_PORT=22 \
/bin/bash "$VENV_HARNESS" >"$WORK/venv.out" 2>"$WORK/venv.err"
vrc=$?
set -e

if [ "$vrc" -eq 0 ]; then
  ok "rebuild branch completes after failed probe"
else
  no "rebuild branch should complete" "rc=$vrc; $(cat "$WORK/venv.err" | tr '\n' '|')"
fi
if [ -f "$VENV_SENTINELS/venv_probe.failed" ] &&
   [ -f "$VENV_SENTINELS/probe_failed_rebuilding.done" ] &&
   [ -f "$VENV_SENTINELS/uv_sync.done" ] &&
   [ -f "$VENV_SENTINELS/jax_cuda.done" ] &&
   [ -f "$VENV_SENTINELS/rebuild_done.done" ]; then
  ok "rebuild branch wrote expected sentinels"
else
  no "rebuild branch sentinels missing" "$(ls "$VENV_SENTINELS" 2>/dev/null | tr '\n' ' ')"
fi
if [ ! -f "$VENV_SENTINELS/probe_ok.done" ]; then
  ok "failed probe did not write probe_ok"
else
  no "failed probe must not write probe_ok" "$(ls "$VENV_SENTINELS" | tr '\n' ' ')"
fi
if grep -q '^run --no-sync python -c' "$UV_LOG" &&
   grep -q '^venv --clear$' "$UV_LOG" &&
   grep -q '^sync$' "$UV_LOG" &&
   grep -q '^pip install -U jax\[cuda12\]$' "$UV_LOG"; then
  ok "failed probe ran clear, sync, and CUDA-JAX reinstall"
else
  no "failed probe rebuild command sequence" "$(cat "$UV_LOG" | tr '\n' '|')"
fi
if grep -q 'venv_probe_branch=probe_failed_rebuilding' "$WORK/venv.err" &&
   grep -q 'venv_probe_branch=rebuild_done' "$WORK/venv.err"; then
  ok "rebuild branch emitted grep-friendly log lines"
else
  no "rebuild branch should log branch markers" "$(cat "$WORK/venv.err" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "FIX 1 name-based sweep tears down a pod created with an unparseable result"

# Stub model: `pod create` ALWAYS records the requested --name in a fake pod
# registry (a real pod exists server-side) but prints UNPARSEABLE JSON, so the
# driver never parses a POD_ID. `pod list` reflects the registry. `pod remove`
# deletes from the registry and logs the teardown. The name-based sweep (FIX 1)
# must find every leaked pod by RUN_TAG prefix and remove it, even though
# POD_ID was never set.
F1="$WORK/fix1"; mkdir -p "$F1/stubs"
REG="$F1/registry"          # one "<id> <name>" per line = live pods
TLOG="$F1/teardown.log"
CLOG="$F1/create.log"
: > "$REG"; : > "$TLOG"; : > "$CLOG"

cat > "$F1/stubs/runpodctl" <<STUB
#!/usr/bin/env bash
REG="$REG"; TLOG="$TLOG"; CLOG="$CLOG"; FIX="$FIX"
case "\$1 \$2" in
  "pod create")
    name="?"; prev=""
    for a in "\$@"; do [ "\$prev" = "--name" ] && name="\$a"; prev="\$a"; done
    id="srv-\$(date +%s%N)-\$RANDOM"
    printf '%s %s\n' "\$id" "\$name" >> "\$REG"
    echo "create name=\$name" >> "\$CLOG"
    # Unparseable result: the pod EXISTS server-side but the driver can't parse
    # an id. This is the create-leak the name-based sweep must still catch.
    echo 'NOT JSON AT ALL <<<'
    exit 0
    ;;
  "pod get")
    cat "\$FIX/pod_exited.json" ;;
  "pod list")
    # Emit the live registry as a JSON array of {id,name}.
    printf '['
    first=1
    while read -r id name; do
      [ -z "\$id" ] && continue
      [ \$first -eq 1 ] || printf ','
      printf '{"id":"%s","name":"%s"}' "\$id" "\$name"
      first=0
    done < "\$REG"
    printf ']\n'
    ;;
  "pod remove")
    echo "remove \$3" >> "\$TLOG"
    # Drop it from the registry.
    grep -v "^\$3 " "\$REG" > "\$REG.tmp" 2>/dev/null || true
    mv "\$REG.tmp" "\$REG" 2>/dev/null || true
    ;;
  "pod stop")
    echo "stop \$3" >> "\$TLOG" ;;
  "user --output"|"user -o"|"user ")
    echo '{"clientBalance":50.0}' ;;
  "datacenter list")
    cat "\$FIX/datacenter_list.json" ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$F1/stubs/runpodctl"
for tool in ssh rsync curl; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$F1/stubs/$tool"; chmod +x "$F1/stubs/$tool"
done

set +e
PATH="$F1/stubs:$PATH" \
RUN_TAG="itest-fix1-$$" \
ENDPOINT_GRACE_SECONDS=2 DEAD_STATE_POLL_SECONDS=1 \
ACQUIRE_WALL_CLOCK_CAP_SECONDS=60 \
RUNPOD_GPU_ID="NVIDIA GeForce RTX 5090" \
RUNPOD_NAME="itest-fix1-$$" \
ACQUIRE_LOCK_FILE="$F1/lock" \
SKIP_IMAGE_CHECK=1 \
bash "$DEPLOY_DIR/runpod_deploy.sh" --acquire-only \
  >"$F1/acq.out" 2>"$F1/acq.err"
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  ok "FIX1: acquisition fails when every create returns unparseable JSON (rc=$rc)"
else
  no "FIX1: acquisition should fail" "rc=$rc; $(tail -3 "$F1/acq.err" | tr '\n' '|')"
fi

n_created=$(wc -l < "$CLOG" | tr -d ' ')
if [ "$n_created" -ge 1 ]; then
  ok "FIX1: at least one create attempt recorded ($n_created)"
else
  no "FIX1: expected create attempts" "created=$n_created"
fi

# Every server-side pod must have been swept by name. The registry must be empty
# (all removed) and the teardown log must contain a remove per created pod.
n_torn=$( { grep -c '^remove ' "$TLOG" || true; } | head -1); n_torn=${n_torn:-0}
n_left=$( { grep -c . "$REG" || true; } | head -1); n_left=${n_left:-0}
if [ "$n_torn" -ge "$n_created" ] && [ "$n_left" -eq 0 ]; then
  ok "FIX1: name-based sweep tore down all leaked pods ($n_torn removed, $n_left left)"
else
  no "FIX1: leaked pods must all be swept" \
     "torn=$n_torn created=$n_created left=$n_left; reg=[$(tr '\n' '|' < "$REG")]"
fi

# Sanity: the unique names we passed all started with RUN_TAG, so the sweep's
# prefix match is meaningful (not a match-everything).
if grep -q "create name=itest-fix1-$$" "$CLOG"; then
  ok "FIX1: create used unique RUN_TAG-prefixed names"
else
  no "FIX1: create should use RUN_TAG-prefixed names" "$(cat "$CLOG" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "FIX 2 INT/TERM handler runs cleanup then exits 130/143 (no resume)"

# A subshell sources the driver's cleanup/signal functions and a sentinel that
# would be touched if execution resumed PAST the signal handler. The handler
# must exit before that sentinel is written, with status 130 (INT) / 143 (TERM).
F2="$WORK/fix2"; mkdir -p "$F2"
# Extract the real signal/cleanup handlers from the driver in THIS shell (BSD
# sed handles the `{`-bearing address fine here) and write them to a sourceable
# harness so the subshell does not have to run sed itself.
F2_FNS="$F2/handlers.sh"
{
  sed -n '/^acquire_cleanup() {/,/^}/p' "$DEPLOY_DIR/runpod_deploy.sh"
  sed -n '/^acquire_signal_trap() {/,/^}/p' "$DEPLOY_DIR/runpod_deploy.sh"
} > "$F2_FNS"
for sig in INT TERM; do
  RESUMED="$F2/resumed_$sig"; CLEANED="$F2/cleaned_$sig"
  rm -f "$RESUMED" "$CLEANED"
  case "$sig" in INT) want=130 ;; TERM) want=143 ;; esac
  set +e
  RESUMED="$RESUMED" CLEANED="$CLEANED" SIG="$sig" \
  DEPLOY_DIR="$DEPLOY_DIR" F2_FNS="$F2_FNS" \
  bash -c '
    set -uo pipefail
    source "$DEPLOY_DIR/lib_acquire.sh"
    # Minimal stand-ins for the driver globals/functions acquire_cleanup needs.
    DRY_RUN=0; ACQUIRED=0; POD_ID=""; RUN_TAG=""; ACQUIRE_LOCK_HELD=0
    ACQUIRE_CLEANUP_DONE=0
    log() { :; }
    teardown_pod() { :; }
    acquire_lock_release() { :; }
    # Override sweep so cleanup records a marker we can assert on.
    sweep_created_pods() { : > "$CLEANED"; }
    # Pull the real signal/cleanup handlers out of the driver.
    source "$F2_FNS"
    trap "acquire_signal_trap $SIG" "$SIG"
    # Raise the signal mid-"acquisition", then a line that must NOT run.
    kill -"$SIG" $$
    sleep 5
    : > "$RESUMED"   # reached only if the handler returned instead of exiting
  '
  rc=$?
  set -e
  if [ "$rc" -eq "$want" ]; then
    ok "FIX2: $sig handler exits with $want"
  else
    no "FIX2: $sig handler must exit $want" "rc=$rc"
  fi
  if [ -f "$CLEANED" ]; then
    ok "FIX2: $sig handler ran cleanup"
  else
    no "FIX2: $sig handler must run cleanup" "no cleaned marker"
  fi
  if [ ! -f "$RESUMED" ]; then
    ok "FIX2: $sig did NOT resume past the handler"
  else
    no "FIX2: $sig must not resume execution" "resumed marker present"
  fi
done

# ---------------------------------------------------------------------------
section "FIX 3 MAX_PARALLEL_ROWS is a hard cap under a fast launcher"

# Each row's mock command increments a shared concurrency counter on start and
# decrements on end, recording the running max. Because launch_row reserves the
# slot SYNCHRONOUSLY via the `.started` marker (FIX 3), the observed max
# concurrency must never exceed MAX_PARALLEL_ROWS — even when the launcher is
# fast and the stagger is zero. We test cap=1 and cap=2.
run_cap_test() {
  local cap=$1 nrows=$2
  local CW="$WORK/fix3_cap$cap"; rm -rf "$CW"; mkdir -p "$CW"
  local REMOTE_FS="$CW/remote_fs"
  local SDIR="$REMOTE_FS/feedbax_runs/runpod-deploy/sentinels"
  mkdir -p "$REMOTE_FS/rlrmp"
  local CUR="$CW/cur" MAXF="$CW/max"
  printf '0' > "$CUR"; printf '0' > "$MAXF"

  # ssh stub executes the remote command locally. The concurrency bookkeeping is
  # injected by replacing the row command with a wrapper that bumps the counter.
  local STUBS3="$CW/stubs"; mkdir -p "$STUBS3"
  cat > "$STUBS3/ssh" <<'STUB'
#!/usr/bin/env bash
cmd="${!#}"
bash -c "$cmd"
STUB
  chmod +x "$STUBS3/ssh"
  for tool in rsync curl runpodctl; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$STUBS3/$tool"; chmod +x "$STUBS3/$tool"
  done

  # A row command that records concurrency. Uses a flock-free atomic-ish bump
  # via a lock dir so the shared counter stays consistent under parallelism.
  local BUMP="$CW/bump.sh"
  cat > "$BUMP" <<BUMPSTUB
#!/usr/bin/env bash
CUR="$CUR"; MAXF="$MAXF"; LOCK="$CW/lock.d"
acquire() { while ! mkdir "\$LOCK" 2>/dev/null; do :; done; }
release() { rmdir "\$LOCK" 2>/dev/null || true; }
acquire; n=\$(cat "\$CUR"); n=\$((n+1)); printf '%s' "\$n" > "\$CUR"
m=\$(cat "\$MAXF"); [ "\$n" -gt "\$m" ] && printf '%s' "\$n" > "\$MAXF"; release
sleep 0.5
acquire; n=\$(cat "\$CUR"); n=\$((n-1)); printf '%s' "\$n" > "\$CUR"; release
BUMPSTUB
  chmod +x "$BUMP"

  # Build a manifest of N rows, each running the bump script.
  local rows_json="$CW/rows.json"
  {
    printf '{"schema_version":1,"rows":['
    local i first=1
    for i in $(seq 1 "$nrows"); do
      [ "$first" -eq 1 ] || printf ','
      printf '{"id":"row_%s","command":"bash %s"}' "$i" "$BUMP"
      first=0
    done
    printf ']}'
  } > "$rows_json"

  local HN="$CW/harness.sh"
  {
    echo 'set -uo pipefail'
    echo "source '$DEPLOY_DIR/lib_acquire.sh'"
    echo "source '$DEPLOY_DIR/lib_run_prep.sh'"
    cat <<'H'
DRY_RUN=0
RUNPOD_SSH_KEY="/dev/null"
SSH_CONNECT_TIMEOUT=10
log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
print_cmd() { :; }
run_cmd() { "$@"; }
rsync_rsh() { printf 'ssh\n'; }
sq() { local v=${1-}; v=${v//\'/\'\\\'\'}; printf "'%s'" "$v"; }
expand_path() { printf '%s\n' "$1"; }
remote_cmd() { ssh -i /dev/null -p 22 root@localhost "$1"; }
remote_capture() { ssh -i /dev/null -p 22 root@localhost "$1"; }
H
    echo 'launch_training'
  } > "$HN"

  set +e
  PATH="$STUBS3:$PATH" \
  REMOTE_RLRMP_ROOT="$REMOTE_FS/rlrmp" \
  REMOTE_RUN_DIR="$REMOTE_FS/feedbax_runs/runpod-deploy" \
  REMOTE_SENTINEL_DIR="$SDIR" \
  JAX_COMPILATION_CACHE_DIR="$REMOTE_FS/jax_cache" \
  ROWS_MANIFEST="$rows_json" \
  ROW_LAUNCH_STAGGER_SECONDS=0 \
  MAX_PARALLEL_ROWS="$cap" \
  SENTINEL_POLL_SECONDS=1 \
  SSH_HOST=localhost SSH_PORT=22 \
  bash "$HN" >"$CW/out" 2>"$CW/err"
  set -e
  # Let any in-flight bumps drain.
  sleep 1
  local observed
  observed=$(cat "$MAXF")
  if [ "$observed" -le "$cap" ] && [ "$observed" -ge 1 ]; then
    ok "FIX3: cap=$cap, $nrows rows -> max concurrency $observed <= $cap"
  else
    no "FIX3: cap=$cap must bound concurrency" "observed max=$observed cap=$cap; $(tail -3 "$CW/err" | tr '\n' '|')"
  fi
}

run_cap_test 1 4
run_cap_test 2 5

# ---------------------------------------------------------------------------
section "FIX 5 balance precheck fails CLOSED when balance unreadable"

# Stub `runpodctl user` to FAIL (nonzero). The driver must refuse to create a
# pod (non-zero exit, no `pod create`) rather than continuing past the guard.
F5="$WORK/fix5"; mkdir -p "$F5/stubs"
CLOG5="$F5/create.log"; : > "$CLOG5"
cat > "$F5/stubs/runpodctl" <<STUB
#!/usr/bin/env bash
case "\$1 \$2" in
  "pod create") echo "create" >> "$CLOG5"; echo '{"id":"should-not-happen"}' ;;
  "user --output"|"user -o"|"user ") echo 'transport error' >&2; exit 7 ;;
  "datacenter list") cat "$FIX/datacenter_list.json" ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$F5/stubs/runpodctl"
for tool in ssh rsync curl; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$F5/stubs/$tool"; chmod +x "$F5/stubs/$tool"
done

set +e
PATH="$F5/stubs:$PATH" \
RUN_TAG="itest-fix5-$$" \
RUNPOD_GPU_ID="NVIDIA GeForce RTX 5090" \
RUNPOD_NAME="itest-fix5-$$" \
ACQUIRE_LOCK_FILE="$F5/lock" \
SKIP_IMAGE_CHECK=1 \
bash "$DEPLOY_DIR/runpod_deploy.sh" --acquire-only \
  >"$F5/out" 2>"$F5/err"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  ok "FIX5: refuses (rc=$rc) when balance is unreadable"
else
  no "FIX5: must fail closed" "rc=$rc"
fi
n_create5=$( { grep -c '^create' "$CLOG5" || true; } | head -1); n_create5=${n_create5:-0}
if [ "$n_create5" -eq 0 ]; then
  ok "FIX5: no pod was created (guard held before create)"
else
  no "FIX5: must not create a pod" "creates=$n_create5"
fi
if grep -q 'balance could not be read' "$F5/err"; then
  ok "FIX5: emitted fail-closed reason"
else
  no "FIX5: should explain fail-closed" "$(tail -3 "$F5/err" | tr '\n' '|')"
fi

# Opt-out: ALLOW_UNKNOWN_BALANCE=1 should NOT block on an unreadable balance.
# (It will still fail later for other reasons, but the balance guard must pass.)
set +e
PATH="$F5/stubs:$PATH" \
RUN_TAG="itest-fix5b-$$" \
ALLOW_UNKNOWN_BALANCE=1 \
ENDPOINT_GRACE_SECONDS=2 DEAD_STATE_POLL_SECONDS=1 \
ACQUIRE_WALL_CLOCK_CAP_SECONDS=20 \
RUNPOD_GPU_ID="NVIDIA GeForce RTX 5090" \
RUNPOD_NAME="itest-fix5b-$$" \
ACQUIRE_LOCK_FILE="$F5/lockb" \
SKIP_IMAGE_CHECK=1 \
bash "$DEPLOY_DIR/runpod_deploy.sh" --acquire-only \
  >"$F5/outb" 2>"$F5/errb"
set -e
if grep -q 'ALLOW_UNKNOWN_BALANCE=1, continuing' "$F5/errb"; then
  ok "FIX5: ALLOW_UNKNOWN_BALANCE=1 bypasses the balance guard"
else
  no "FIX5: opt-out should bypass guard" "$(grep -i balance "$F5/errb" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "W8 declared baseline preflight fails before launch and stages valid baseline"

W8="$WORK/w8"; mkdir -p "$W8/rlrmp" "$W8/jax"
cat > "$W8/spec_missing.json" <<JSON
{"user_confirmed":true,"resume":{"baseline_checkpoint":"_artifacts/run/checkpoints","completed_batches":42}}
JSON

set +e
RLRMP_ROOT="$W8/rlrmp" \
JAX_COOKBOOK_ROOT="$W8/jax" \
RUNPOD_RUN_CONFIG_FILE="$W8/run-config.json" \
SKIP_IMAGE_CHECK=1 \
bash "$DEPLOY_DIR/runpod_deploy.sh" --dry-run --pod-id pod-w8 --ssh-host dry --ssh-port 22 \
  --train-spec "$W8/spec_missing.json" --launch-command true \
  >"$W8/missing.out" 2>"$W8/missing.err"
rc=$?
set -e
if [ "$rc" -ne 0 ] && grep -q 'baseline preflight failed: source checkpoint not found' "$W8/missing.err"; then
  ok "W8: missing declared baseline fails before launch"
else
  no "W8: missing baseline should fail specifically" "rc=$rc; $(cat "$W8/missing.err" | tr '\n' '|')"
fi
if ! grep -q 'launching row training' "$W8/missing.err"; then
  ok "W8: missing baseline stops before training launch"
else
  no "W8: missing baseline must not launch" "$(cat "$W8/missing.err" | tr '\n' '|')"
fi

mkdir -p "$W8/rlrmp/_artifacts/run/checkpoints"
cat > "$W8/rlrmp/_artifacts/run/checkpoints/latest.json" <<JSON
{"completed_coordinate":{"global_step":42}}
JSON
cat > "$W8/spec_ok.json" <<JSON
{"user_confirmed":true,"resume":{"baseline_checkpoint":"_artifacts/run/checkpoints","completed_batches":42}}
JSON
set +e
RLRMP_ROOT="$W8/rlrmp" \
JAX_COOKBOOK_ROOT="$W8/jax" \
RUNPOD_RUN_CONFIG_FILE="$W8/run-config.json" \
REMOTE_DEPLOY_CONFIG_PATH="/workspace/feedbax_runs/latest-run-config.json" \
SKIP_IMAGE_CHECK=1 \
bash "$DEPLOY_DIR/runpod_deploy.sh" --dry-run --pod-id pod-w8 --ssh-host dry --ssh-port 22 \
  --train-spec "$W8/spec_ok.json" --launch-command true \
  >"$W8/ok.out" 2>"$W8/ok.err"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  ok "W8: valid declared baseline passes dry-run deploy"
else
  no "W8: valid baseline dry-run should pass" "rc=$rc; $(tail -10 "$W8/ok.err" | tr '\n' '|')"
fi
if cat "$W8/ok.out" "$W8/ok.err" | grep -q 'staging declared baseline' &&
   cat "$W8/ok.out" "$W8/ok.err" | grep -q '_artifacts/run/checkpoints' &&
   cat "$W8/ok.out" "$W8/ok.err" | grep -q 'latest-run-config.json'; then
  ok "W8: valid baseline stages and syncs fixed deploy config"
else
  no "W8: valid baseline should stage baseline and config" "$(cat "$W8/ok.out" "$W8/ok.err" | grep -E 'baseline|run-config|rsync' | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "W8 poll_run derives run dir from deploy config and refuses silent default"

POLL="$WORK/poll"; mkdir -p "$POLL/stubs" "$POLL/remote/run/sentinels" "$POLL/remote/run/logs"
CONFIG="$POLL/deploy-config.json"
cat > "$CONFIG" <<JSON
{"schema_version":1,"remote_run_dir":"$POLL/remote/run","remote_sentinel_dir":"$POLL/remote/run/sentinels","remote_checkpoint_dir":"$POLL/remote/run","remote_log_dir":"$POLL/remote/run/logs"}
JSON
printf '%s\n' "$$" > "$POLL/remote/run/sentinels/row_live.pid"
: > "$POLL/remote/run/sentinels/row_stale.started"

cat > "$POLL/stubs/runpodctl" <<'STUB'
#!/usr/bin/env bash
case "$1 $2" in
  "pod get") echo '{"desiredStatus":"RUNNING","ssh":{"ip":"localhost","port":22}}' ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$POLL/stubs/runpodctl"
cat > "$POLL/stubs/ssh" <<'STUB'
#!/usr/bin/env bash
cmd="${!#}"
bash -c "$cmd"
STUB
chmod +x "$POLL/stubs/ssh"

set +e
PATH="$POLL/stubs:$PATH" \
REMOTE_DEPLOY_CONFIG_PATH="$CONFIG" \
bash "$DEPLOY_DIR/poll_run.sh" --pod-id pod-poll --cadence-seconds 0 \
  >"$POLL/out" 2>"$POLL/err"
rc=$?
set -e
if [ "$rc" -eq 0 ] &&
   grep -q 'rows_running=1 rows_stale=1' "$POLL/out" &&
   grep -q 'row_stale:stale_started' "$POLL/out"; then
  ok "W8: poll derives run dir and distinguishes stale started rows"
else
  no "W8: poll should derive config and report stale" "rc=$rc; out=[$(cat "$POLL/out" | tr '\n' '|')] err=[$(cat "$POLL/err" | tr '\n' '|')]"
fi

set +e
PATH="$POLL/stubs:$PATH" \
REMOTE_DEPLOY_CONFIG_PATH="$POLL/missing-config.json" \
bash "$DEPLOY_DIR/poll_run.sh" --pod-id pod-poll --cadence-seconds 0 \
  >"$POLL/missing.out" 2>"$POLL/missing.err"
rc=$?
set -e
if [ "$rc" -ne 0 ] && grep -q 'remote run dir is not set' "$POLL/missing.err"; then
  ok "W8: poll refuses missing run dir/config instead of defaulting"
else
  no "W8: poll should refuse missing run dir/config" "rc=$rc; $(cat "$POLL/missing.err" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
section "W8 row signal trap writes failed sentinel"

SIG="$WORK/signal"; mkdir -p "$SIG/stubs" "$SIG/remote/rlrmp"
SIG_SENT="$SIG/remote/run/sentinels"
cat > "$SIG/stubs/ssh" <<'STUB'
#!/usr/bin/env bash
cmd="${!#}"
bash -c "$cmd"
STUB
chmod +x "$SIG/stubs/ssh"

SIG_HARNESS="$SIG/harness.sh"
{
  echo 'set -uo pipefail'
  echo "source '$DEPLOY_DIR/lib_acquire.sh'"
  echo "source '$DEPLOY_DIR/lib_run_prep.sh'"
  cat <<'H'
DRY_RUN=0
RUNPOD_SSH_KEY="/dev/null"
SSH_CONNECT_TIMEOUT=10
log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
sq() { local v=${1-}; v=${v//\'/\'\\\'\'}; printf "'%s'" "$v"; }
remote_cmd() { ssh -i /dev/null -p 22 root@localhost "$1"; }
H
  echo 'launch_row "row_term" "$REMOTE_RLRMP_ROOT" "sleep 20"'
} > "$SIG_HARNESS"

PATH="$SIG/stubs:$PATH" \
REMOTE_RLRMP_ROOT="$SIG/remote/rlrmp" \
REMOTE_RUN_DIR="$SIG/remote/run" \
REMOTE_SENTINEL_DIR="$SIG_SENT" \
JAX_COMPILATION_CACHE_DIR="$SIG/remote/jax_cache" \
SSH_HOST=localhost SSH_PORT=22 \
bash "$SIG_HARNESS" >"$SIG/out" 2>"$SIG/err"
sleep 1
pid=$(cat "$SIG_SENT/row_term.pid" 2>/dev/null || true)
if [ -n "$pid" ]; then
  kill -TERM "$pid" 2>/dev/null || true
fi
sleep 1
if [ -f "$SIG_SENT/row_term.failed" ] && [ ! -f "$SIG_SENT/row_term.done" ]; then
  ok "W8: TERM writes .failed and not .done"
else
  no "W8: TERM should leave failed sentinel" "pid=${pid:-none}; files=[$(find "$SIG_SENT" -maxdepth 1 -type f -print 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')]"
fi

# ---------------------------------------------------------------------------
printf '\n== summary ==\n'
printf 'PASS=%s FAIL=%s\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
