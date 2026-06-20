#!/usr/bin/env bash
# Plain-bash unit tests for scripts/deploy/lib_acquire.sh and the driver
# dry-run paths. No bats / shellcheck available in this environment, so this is
# a minimal self-contained assertion harness.
#
#   bash scripts/deploy/tests/run_tests.sh
#
# Exit 0 = all pass.
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEPLOY_DIR="$(cd "$TESTS_DIR/.." && pwd -P)"
FIX="$TESTS_DIR/fixtures"

# shellcheck source=../lib_acquire.sh
source "$DEPLOY_DIR/lib_acquire.sh"

PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    printf '  ok   %s\n' "$1"
}
no() {
    FAIL=$((FAIL + 1))
    printf '  FAIL %s\n    expected: %s\n    actual:   %s\n' "$1" "$2" "$3"
}

eq() {
    # eq <label> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1"; else no "$1" "$2" "$3"; fi
}

section() { printf '\n== %s ==\n' "$1"; }

# ---------------------------------------------------------------------------
section "W3 classify_pod_state"
out=$(classify_pod_state < "$FIX/pod_ready.json"); rc=$?
eq "ready -> stdout" "ready 213.173.108.5 39125" "$out"
eq "ready -> rc 0" "0" "$rc"

out=$(classify_pod_state < "$FIX/pod_booting.json"); rc=$?
eq "booting -> not_ready" "not_ready" "$out"
eq "booting -> rc 2" "2" "$rc"

out=$(classify_pod_state < "$FIX/pod_exited.json"); rc=$?
eq "exited -> dead" "dead Exited_by_Runpod:_image_pull_failed" "$out"
eq "exited -> rc 3" "3" "$rc"

out=$(classify_pod_state < "$FIX/pod_user_stopped.json"); rc=$?
eq "user-stop -> dead" "dead Exited_by_user:_2026-06-20T10:05:00Z" "$out"
eq "user-stop -> rc 3" "3" "$rc"

out=$(printf 'not json' | classify_pod_state); rc=$?
eq "garbage -> unknown" "unknown" "$out"
eq "garbage -> rc 4" "4" "$rc"

# ssh_command-only ready form (no .ssh.ip / .ssh.port)
out=$(printf '%s' '{"desiredStatus":"RUNNING","ssh":{"ssh_command":"ssh -i k root@1.2.3.4 -p 222"}}' \
    | classify_pod_state)
eq "ssh_command ready form" "ready 1.2.3.4 222" "$out"

# No publicIp fallback: a secure pod with publicIp but no .ssh.ip is NOT ready.
out=$(printf '%s' '{"desiredStatus":"RUNNING","publicIp":"9.9.9.9","ssh":{"error":"pod not ready"}}' \
    | classify_pod_state)
eq "publicIp is NOT a fallback" "not_ready" "$out"

# ---------------------------------------------------------------------------
section "W1 rank_datacenters_for_gpu"
out=$(rank_datacenters_for_gpu "NVIDIA GeForce RTX 5090" < "$FIX/datacenter_list.json" | tr '\n' ',')
# EU-RO-1 High, US-KS-2 Medium, AP-IN-1 Low; EU-SE-1 "" excluded; CA-MTL-1 null skipped.
eq "5090 ranked best-first" "EU-RO-1,US-KS-2,AP-IN-1," "$out"

out=$(rank_datacenters_for_gpu "NVIDIA GeForce RTX 4090" < "$FIX/datacenter_list.json" | tr '\n' ',')
eq "4090 ranked best-first" "US-TX-3,EU-RO-1," "$out"

out=$(rank_datacenters_for_gpu "NVIDIA NONEXISTENT" < "$FIX/datacenter_list.json" | tr '\n' ',')
eq "unknown gpu -> empty" "" "$out"

# ---------------------------------------------------------------------------
section "W1 check_balance_floor"
out=$(printf '%s' '{"clientBalance":5.97}' | check_balance_floor 1.0); rc=$?
eq "above floor -> ok" "ok 5.97" "$out"
eq "above floor -> rc 0" "0" "$rc"

out=$(printf '%s' '{"clientBalance":0.5}' | check_balance_floor 1.0); rc=$?
eq "below floor -> low" "low 0.5 1.0" "$out"
eq "below floor -> rc 1" "1" "$rc"

out=$(printf '%s' '{}' | check_balance_floor 1.0); rc=$?
eq "no balance -> unknown" "unknown" "$out"
eq "no balance -> rc 2" "2" "$rc"

# ---------------------------------------------------------------------------
section "W2 rows manifest"
manifest='{"schema_version":1,"rows":[
  {"id":"a","command":"sleep 1"},
  {"id":"b","command":"sleep 1"},
  {"id":"c","command":"sleep 1"}]}'
out=$(printf '%s' "$manifest" | validate_rows_manifest); rc=$?
eq "valid manifest -> ok 3" "ok 3" "$out"
eq "valid manifest -> rc 0" "0" "$rc"

out=$(printf '%s' "$manifest" | rows_manifest_ids | tr '\n' ',')
eq "manifest ids" "a,b,c," "$out"

out=$(printf '%s' "$manifest" | rows_manifest_field b command)
eq "manifest field" "sleep 1" "$out"

out=$(printf '%s' '{"schema_version":1,"rows":[]}' | validate_rows_manifest); rc=$?
eq "empty rows -> error" "error no_rows" "$out"

out=$(printf '%s' '{"schema_version":1,"rows":[{"id":"a","command":"x"},{"id":"a","command":"y"}]}' \
    | validate_rows_manifest)
eq "dup id -> error" "error duplicate_row_id" "$out"

out=$(printf '%s' '{"schema_version":99,"rows":[{"id":"a","command":"x"}]}' | validate_rows_manifest)
eq "bad version -> error" "error schema_version_expected_1_got_99" "$out"

out=$(printf '%s' '{"rows":[{"id":"a"}]}' | validate_rows_manifest)
eq "missing command -> error" "error schema_version_expected_1_got_0" "$out"

# ---------------------------------------------------------------------------
section "W7 _artifacts symlink repair"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# dangling symlink
ln -s "$TMP/nonexistent_target" "$TMP/dangling"
eq "classify dangling" "dangling_symlink" "$(classify_artifacts_path "$TMP/dangling")"
out=$(repair_artifacts_symlink "$TMP/dangling")
eq "repair dangling -> repaired" "repaired dangling_symlink" "$out"
eq "dangling now real dir" "real_dir" "$(classify_artifacts_path "$TMP/dangling")"
# idempotent
out=$(repair_artifacts_symlink "$TMP/dangling")
eq "repair idempotent -> left real_dir" "left real_dir" "$out"

# real dir
mkdir -p "$TMP/realdir"
eq "classify real_dir" "real_dir" "$(classify_artifacts_path "$TMP/realdir")"
out=$(repair_artifacts_symlink "$TMP/realdir")
eq "repair real_dir -> left" "left real_dir" "$out"

# valid symlink
mkdir -p "$TMP/target"
ln -s "$TMP/target" "$TMP/valid"
eq "classify valid_symlink" "valid_symlink" "$(classify_artifacts_path "$TMP/valid")"
out=$(repair_artifacts_symlink "$TMP/valid")
eq "repair valid -> warned" "warned valid_symlink" "$out"
eq "valid symlink preserved" "valid_symlink" "$(classify_artifacts_path "$TMP/valid")"

# missing
eq "classify missing" "missing" "$(classify_artifacts_path "$TMP/nope")"
out=$(repair_artifacts_symlink "$TMP/nope")
eq "repair missing -> left" "left missing" "$out"

# ---------------------------------------------------------------------------
section "W4 aggregate progress"
out=$(printf 'a done\nb running\nc failed\nd pending\ne done\n' | rows_rollup_from_listing)
eq "rows rollup" "rows_done=2 rows_failed=1 rows_running=1 rows_pending=1 rows_total=5" "$out"

# checkpoint dirs
mkdir -p "$TMP/ckpt/checkpoint_100" "$TMP/ckpt/checkpoint_1200" "$TMP/ckpt/checkpoint_50"
eq "max checkpoint step" "1200" "$(max_checkpoint_step "$TMP/ckpt")"
eq "max checkpoint empty -> none" "none" "$(max_checkpoint_step "$TMP/empty")"

printf 'compiling...\nbatch 10 loss 5.0\nbatch 1234 loss 0.3\n' > "$TMP/train.log"
eq "last batch from log" "1234" "$(last_batch_from_log "$TMP/train.log")"
printf 'compiling...\nno counter yet\n' > "$TMP/quiet.log"
eq "no batch -> none" "none" "$(last_batch_from_log "$TMP/quiet.log")"

# ---------------------------------------------------------------------------
section "W4 progress_report over a mock remote layout"
# Mock remote layout: 3 rows (a done, b running, c pending-only), a checkpoint
# dir, and per-row logs.
RR="$TMP/remote"
SDIR="$RR/sentinels"
CDIR="$RR"
LDIR="$RR/logs"
mkdir -p "$SDIR" "$LDIR" "$CDIR/checkpoint_800" "$CDIR/checkpoint_2400"
# bootstrap sentinels should be filtered out of the rows roll-up
: > "$SDIR/uv_sync.done"
: > "$SDIR/jax_cuda.done"
# row a: done
: > "$SDIR/row_a.pid"; : > "$SDIR/row_a.done"
# row b: running (pid, no terminal)
: > "$SDIR/row_b.pid"
# row c: failed
: > "$SDIR/row_c.pid"; : > "$SDIR/row_c.failed"
printf 'compiling\nbatch 100\nbatch 2350 loss 0.4\n' > "$LDIR/row_a.log"
printf 'compiling only\n' > "$LDIR/row_b.log"

eq "discover_row_ids (rows only, sorted, bootstrap excluded)" \
   "row_a,row_b,row_c," \
   "$(discover_row_ids "$SDIR" | grep -vE '^(uv_sync|jax_cuda)$' | tr '\n' ',')"
eq "row_state done" "done" "$(row_state "$SDIR" row_a)"
eq "row_state running" "running" "$(row_state "$SDIR" row_b)"
eq "row_state failed" "failed" "$(row_state "$SDIR" row_c)"

report=$(progress_report "$SDIR" "$CDIR" "$LDIR")
# progress_report does NOT filter bootstrap sentinels (it counts every marker);
# the remote command does the filtering. Verify last_checkpoint and last_batch
# which are the progress signals the test targets.
case "$report" in
  *"last_checkpoint=2400"*) ok "progress_report last_checkpoint=2400" ;;
  *) no "progress_report last_checkpoint=2400" "...last_checkpoint=2400..." "$report" ;;
esac
case "$report" in
  *"last_batch=2350"*) ok "progress_report last_batch=2350" ;;
  *) no "progress_report last_batch=2350" "...last_batch=2350..." "$report" ;;
esac

# End-to-end: render the remote-status command and run it locally against the
# mock layout. This exercises the EXACT string the pod runs over SSH, including
# bootstrap-sentinel filtering and the rows roll-up.
# shellcheck source=../poll_run.sh
# We can't source poll_run.sh (it runs main); instead extract the function.
REMOTE_STATUS_FN=$(
  sed -n '/^build_remote_status_command() {/,/^}/p' "$DEPLOY_DIR/poll_run.sh"
)
eval "$REMOTE_STATUS_FN"
remote_cmd=$(build_remote_status_command "$SDIR" "$CDIR" "$LDIR")
rendered=$(bash -c "$remote_cmd" 2>/dev/null)
case "$rendered" in
  *"rows_done=1 rows_failed=1 rows_running=1 rows_pending=0 rows_total=3"*)
    ok "remote status rows roll-up (bootstrap filtered)" ;;
  *) no "remote status rows roll-up" \
        "...rows_done=1 rows_failed=1 rows_running=1 rows_pending=0 rows_total=3..." \
        "$rendered" ;;
esac
case "$rendered" in
  *"uv_sync=done"*) ok "remote status uv_sync=done" ;;
  *) no "remote status uv_sync=done" "...uv_sync=done..." "$rendered" ;;
esac
case "$rendered" in
  *"last_checkpoint=2400 last_batch=2350"*) ok "remote status progress signals" ;;
  *) no "remote status progress signals" "...last_checkpoint=2400 last_batch=2350..." "$rendered" ;;
esac

# ---------------------------------------------------------------------------
printf '\n== summary ==\n'
printf 'PASS=%s FAIL=%s\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
