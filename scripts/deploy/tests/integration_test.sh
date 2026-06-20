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

# Drive launch_training directly via a tiny harness that sources the script's
# functions. We can't `source` runpod_deploy.sh (it calls main), so we extract
# the launch-related functions and the few helpers they need.
HARNESS="$WORK/launch_harness.sh"
{
  echo 'set -uo pipefail'
  echo "source '$DEPLOY_DIR/lib_acquire.sh'"
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
  # Extract launch_row, count_running_rows, launch_training, sync_rows_manifest.
  for fn in sync_rows_manifest launch_row count_running_rows launch_training; do
    sed -n "/^$fn() {/,/^}/p" "$DEPLOY_DIR/runpod_deploy.sh"
  done
  echo 'launch_training'
} > "$HARNESS"

set +e
PATH="$STUBS2:$PATH" \
REMOTE_RLRMP_ROOT="$REMOTE_FS/rlrmp" \
REMOTE_RUN_DIR="$REMOTE_FS/feedbax_runs/runpod-deploy" \
REMOTE_SENTINEL_DIR="$SDIR2" \
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

if grep -q 'launched 3 row(s)' "$WORK/launch.err"; then
  ok "launch_training reported 3 rows launched"
else
  no "launch_training should report 3 rows" "$(tail -3 "$WORK/launch.err" | tr '\n' '|')"
fi

# ---------------------------------------------------------------------------
printf '\n== summary ==\n'
printf 'PASS=%s FAIL=%s\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
