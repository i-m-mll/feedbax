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
  for fn in remote_nohup_sentinel launch_row; do
    sed -n "/^$fn() {/,/^}/p" "$DEPLOY_DIR/runpod_deploy.sh"
  done
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
    for fn in sync_rows_manifest launch_row count_running_rows launch_training; do
      sed -n "/^$fn() {/,/^}/p" "$DEPLOY_DIR/runpod_deploy.sh"
    done
    echo 'launch_training'
  } > "$HN"

  set +e
  PATH="$STUBS3:$PATH" \
  REMOTE_RLRMP_ROOT="$REMOTE_FS/rlrmp" \
  REMOTE_RUN_DIR="$REMOTE_FS/feedbax_runs/runpod-deploy" \
  REMOTE_SENTINEL_DIR="$SDIR" \
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
printf '\n== summary ==\n'
printf 'PASS=%s FAIL=%s\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
