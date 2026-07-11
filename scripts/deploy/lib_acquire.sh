#!/usr/bin/env bash
# Pure, sourceable helpers for deterministic RunPod acquisition, endpoint
# classification, datacenter ranking, parallel row launch, _artifacts symlink
# repair, and aggregate progress parsing.
#
# Every function here is side-effect-light and unit-testable: it takes JSON or
# strings on stdin/args and writes a machine-readable result to stdout. Network
# and pod-mutating actions live in the driver scripts, not here.
#
# Requires: jq, bash >= 4 (associative-array-free; we stay portable).

# read_lines_into <array_name>  (lines on stdin)
# Portable `mapfile -t` replacement (macOS ships bash 3.2 with no mapfile). Reads
# stdin into the named array, one element per line, preserving empty trailing
# lines as a single empty element only when the input had content.
read_lines_into() {
    local __name=$1 __line
    eval "$__name=()"
    while IFS= read -r __line || [ -n "$__line" ]; do
        eval "$__name+=(\"\$__line\")"
    done
}

# ---------------------------------------------------------------------------
# Endpoint / pod-state classification  (W3 - issue fd086c1)
# ---------------------------------------------------------------------------

# Ground truth (real-pod probe 2026-06-20, secure RTX PRO 4500 Blackwell):
#   - The endpoint lives ONLY in the .ssh object on secure pods. publicIp /
#     portMappings are ABSENT on secure cloud; never fall back to them.
#   - Booting:   .ssh = {"error":"pod not ready", ...}  (no .ssh.ip)
#   - Ready:     .ssh = {"ip":"213.x.x.x","port":39125,"ssh_command":"..."}
#   - Dead-state: desiredStatus flips RUNNING -> EXITED/TERMINATED within ~6s
#     for a failed/unpullable image; .ssh.status mirrors it. lastStatusChange
#     reads "Exited by Runpod: ..." or "Exited by user: ...".
#
# classify_pod_state reads pod JSON on stdin and prints exactly one line:
#   ready <ip> <port>     endpoint assigned, usable
#   dead <reason>         desiredStatus EXITED/TERMINATED -> tear down now
#   not_ready             still RUNNING, no endpoint yet (grace window applies)
#   unknown               JSON unparseable / pod object absent
#
# Exit status mirrors the class so callers can branch on $? as well as stdout:
#   0 ready, 2 not_ready, 3 dead, 4 unknown.
classify_pod_state() {
    local detail
    detail=$(cat)
    if ! printf '%s' "$detail" | jq -e . >/dev/null 2>&1; then
        printf 'unknown\n'
        return 4
    fi

    local desired ssh_status
    desired=$(printf '%s' "$detail" | jq -r '(.desiredStatus // .status // "") | ascii_upcase')
    ssh_status=$(printf '%s' "$detail" |
        jq -r '(.ssh | if type == "object" then (.status // "") else "" end) | ascii_upcase')

    # Dead-state fast-fail: caught in seconds, before the long grace window.
    case "$desired" in
        EXITED|TERMINATED|FAILED)
            local reason
            reason=$(printf '%s' "$detail" |
                jq -r '(.lastStatusChange // .ssh.error // "exited")' |
                tr '[:space:]' '_' | tr -cd '[:alnum:]_.:/=@,+-' |
                sed -e 's/_*$//')
            printf 'dead %s\n' "${reason:-exited}"
            return 3
            ;;
    esac
    case "$ssh_status" in
        EXITED|TERMINATED|FAILED)
            printf 'dead ssh_%s\n' "$(printf '%s' "$ssh_status" | tr '[:upper:]' '[:lower:]')"
            return 3
            ;;
    esac

    # Ready: .ssh.ip present (equivalently .ssh.error absent). Also accept the
    # ssh_command form. No publicIp / portMappings fallback on secure cloud.
    local ip port ssh_command
    ip=$(printf '%s' "$detail" |
        jq -r '(.ssh | if type == "object" then (.ip // .host // "") else "" end)')
    port=$(printf '%s' "$detail" |
        jq -r '(.ssh | if type == "object" then (.port // "") else "" end)')

    if [ -z "$ip" ] || [ -z "$port" ]; then
        ssh_command=$(printf '%s' "$detail" |
            jq -r '(.ssh | if type == "object" then (.ssh_command // .sshCommand // .command // "") else "" end)')
        if [ -z "$port" ] && [[ $ssh_command =~ -p[[:space:]]+([0-9]+) ]]; then
            port=${BASH_REMATCH[1]}
        fi
        if [ -z "$ip" ] && [[ $ssh_command =~ root@([^[:space:]]+) ]]; then
            ip=${BASH_REMATCH[1]}
        fi
    fi

    if [ -n "$ip" ] && [ -n "$port" ]; then
        printf 'ready %s %s\n' "$ip" "$port"
        return 0
    fi

    # Still RUNNING, .ssh.error present (or endpoint simply not assigned yet):
    # this is the only case the long grace window is for.
    printf 'not_ready\n'
    return 2
}

# ---------------------------------------------------------------------------
# Datacenter ranking  (W1 - issue c272111)
# ---------------------------------------------------------------------------

# rank_datacenters_for_gpu <gpu_id>  (datacenter-list JSON on stdin)
# Prints candidate DC ids, best stock first, one per line. stockStatus order is
# High > Medium > Low; "" / missing is excluded (no stock). Null gpuAvailability
# entries are skipped safely.
rank_datacenters_for_gpu() {
    local gpu_id=$1
    jq -r --arg gpu "$gpu_id" '
        def rank:
            if . == "High" then 3
            elif . == "Medium" then 2
            elif . == "Low" then 1
            else 0 end;
        [ .[]
          | . as $dc
          | (($dc.gpuAvailability // [])[]
             | select(.gpuId == $gpu)
             | { id: $dc.id, stock: .stockStatus, rank: (.stockStatus | rank) })
        ]
        | map(select(.rank > 0))
        | sort_by(-.rank)
        | .[].id
    '
}

# pod_ids_by_name_prefix <prefix>  (runpodctl `pod list --output json` on stdin)
# Prints the id of every pod whose name starts with <prefix>, one per line.
# Used by the name-based create-leak sweep (FIX 1): the driver tags each create
# attempt with a unique per-run name, then sweeps `pod list` by that name so a
# pod that was created server-side but whose CLI result was nonzero/unparseable
# (or a pod created right before a SIGINT) is still found and torn down even
# when POD_ID was never parsed. Tolerates both a bare array and common wrapper
# shapes (.pods / .data / .data.pods); empty / unparseable input prints nothing.
pod_ids_by_name_prefix() {
    local prefix=$1
    [ -n "$prefix" ] || return 0
    jq -r --arg prefix "$prefix" '
        ( if type == "array" then .
          elif type == "object" then
            (.pods // .data // (.data.pods) // [])
          else [] end )
        | ( if type == "array" then . else [] end )
        | .[]
        | select(type == "object")
        | { id: (.id // .podId // .pod.id // empty),
            name: (.name // .pod.name // "") }
        | select(.id != null and .id != "")
        | select(.name | startswith($prefix))
        | .id
    ' 2>/dev/null || true
}

# validate_row_id <id>  (FIX 4)
# Row ids are interpolated into remote sentinel/log/pid paths and split as shell
# words in poll_run.sh, so they must be constrained to a safe character class.
# Prints "ok" rc 0 for a safe id; "error <reason>" rc 1 otherwise.
validate_row_id() {
    local id=$1 stripped
    if [ -z "$id" ]; then
        printf 'error empty_row_id\n'
        return 1
    fi
    # Reject anything outside [A-Za-z0-9_.-]. Strip the allowed class with tr and
    # compare LENGTHS (not contents) so an embedded newline — which command
    # substitution would otherwise trim away — is still detected. A safe id
    # consists entirely of allowed characters, so stripping leaves length 0.
    stripped=$(printf '%s' "$id" | tr -d 'A-Za-z0-9_.-' | wc -c | tr -d ' ')
    if [ "$stripped" != "0" ]; then
        printf 'error unsafe_row_id\n'
        return 1
    fi
    printf 'ok\n'
    return 0
}

# ---------------------------------------------------------------------------
# Balance precheck  (W1 - issue c272111)
# ---------------------------------------------------------------------------

# check_balance_floor <min_usd>  (runpodctl user JSON on stdin)
# Prints "ok <balance>" and returns 0 if clientBalance >= min; otherwise prints
# "low <balance> <min>" and returns 1. Non-numeric balance -> "unknown", rc 2.
check_balance_floor() {
    local min=$1 balance
    balance=$(jq -r '(.clientBalance // .balance // empty)')
    if [ -z "$balance" ]; then
        printf 'unknown\n'
        return 2
    fi
    # Float comparison via awk (bash can't do floats).
    if awk -v b="$balance" -v m="$min" 'BEGIN { exit !(b + 0 >= m + 0) }'; then
        printf 'ok %s\n' "$balance"
        return 0
    fi
    printf 'low %s %s\n' "$balance" "$min"
    return 1
}

# ---------------------------------------------------------------------------
# Rows manifest  (W2 - issue 9ca50b1)
# ---------------------------------------------------------------------------

# Rows manifest schema (version 1):
#   {
#     "schema_version": 1,
#     "rows": [
#       { "id": "row_a", "command": "uv run ... ", "workdir": "/workspace/rlrmp" },
#       ...
#     ]
#   }
# workdir is optional (defaults applied by the driver). Sentinel/log/pid paths
# are DERIVED from the row id under the sentinel dir, so the manifest stays
# minimal and the driver controls layout.
ROWS_MANIFEST_SCHEMA_VERSION=1

# validate_rows_manifest  (manifest JSON on stdin)
# Prints "ok <n_rows>" rc 0, or "error <message>" rc 1.
validate_rows_manifest() {
    local manifest
    manifest=$(cat)
    if ! printf '%s' "$manifest" | jq -e . >/dev/null 2>&1; then
        printf 'error invalid_json\n'
        return 1
    fi
    local version
    version=$(printf '%s' "$manifest" | jq -r '(.schema_version // 0)')
    if [ "$version" != "$ROWS_MANIFEST_SCHEMA_VERSION" ]; then
        printf 'error schema_version_expected_%s_got_%s\n' \
            "$ROWS_MANIFEST_SCHEMA_VERSION" "$version"
        return 1
    fi
    # Every row needs a non-empty id and command; ids must be unique and safe.
    # Row ids are interpolated into remote sentinel/log/pid paths and split as
    # shell words in poll, so constrain them to [A-Za-z0-9_.-] (FIX 4).
    local problem
    problem=$(printf '%s' "$manifest" | jq -r '
        (.rows // []) as $rows
        | if ($rows | length) == 0 then "no_rows"
          elif ($rows | map(select((.id // "") == "")) | length) > 0 then "row_missing_id"
          elif ($rows | map(select((.command // "") == "")) | length) > 0 then "row_missing_command"
          elif (($rows | map(.id) | unique | length) != ($rows | length)) then "duplicate_row_id"
          elif ($rows | map(select((.id | tostring | test("^[A-Za-z0-9_.-]+$") | not))) | length) > 0 then "unsafe_row_id"
          else "ok" end
    ')
    if [ "$problem" != "ok" ]; then
        printf 'error %s\n' "$problem"
        return 1
    fi
    printf 'ok %s\n' "$(printf '%s' "$manifest" | jq -r '.rows | length')"
    return 0
}

# rows_manifest_ids  (manifest JSON on stdin) -> one id per line, in order.
rows_manifest_ids() {
    jq -r '.rows[].id'
}

# rows_manifest_field <id> <field>  (manifest JSON on stdin) -> field value.
rows_manifest_field() {
    local id=$1 field=$2
    jq -r --arg id "$id" --arg f "$field" '
        .rows[] | select(.id == $id) | (.[$f] // "")
    '
}

# latest_pointer_completed_batches <latest.json>
# Supports current Feedbax checkpoint custody pointers and older RLRMP indexes.
# Coordinates are custody/barrier positions, never training-batch fallbacks.
latest_pointer_completed_batches() {
    local latest_json=$1
    [ -f "$latest_json" ] || return 1
    jq -r '
        .completed_training_batches
        // .completed_batches
        // .completed_batch
        // .completedBatch
        // .metadata.completed_training_batches
        // .metadata.completed_batches
        // .metadata.completed_batch
        // .metadata.completedBatch
        // empty
    ' "$latest_json" 2>/dev/null
}

# ---------------------------------------------------------------------------
# _artifacts symlink repair  (W7 - issue 02c2854)
# ---------------------------------------------------------------------------

# classify_artifacts_path <path>  -> one word describing the local FS state:
#   dangling_symlink   symlink whose target does not exist  (-> repair)
#   valid_symlink      symlink resolving to an existing dir  (-> warn + leave)
#   real_dir           a real directory                      (-> leave)
#   real_file          a real (non-dir) file                 (-> warn + leave)
#   missing            nothing there                         (-> driver mkdir -p)
classify_artifacts_path() {
    local path=$1
    if [ -L "$path" ]; then
        if [ -e "$path" ]; then
            printf 'valid_symlink\n'
        else
            printf 'dangling_symlink\n'
        fi
        return 0
    fi
    if [ -d "$path" ]; then
        printf 'real_dir\n'
        return 0
    fi
    if [ -e "$path" ]; then
        printf 'real_file\n'
        return 0
    fi
    printf 'missing\n'
}

# repair_artifacts_symlink <path>  -> idempotent local repair.
# Replaces a dangling symlink with a real directory; leaves real dirs, valid
# symlinks, and missing paths untouched (driver mkdir -p handles missing).
# Prints "<action> <class>" where action is repaired/left/warned.
# Returns 0 on a safe/handled state, 1 on a real_file collision (caller decides).
repair_artifacts_symlink() {
    local path=$1 class
    class=$(classify_artifacts_path "$path")
    case "$class" in
        dangling_symlink)
            rm -f "$path"
            mkdir -p "$path"
            printf 'repaired %s\n' "$class"
            return 0
            ;;
        real_dir|missing)
            printf 'left %s\n' "$class"
            return 0
            ;;
        valid_symlink)
            printf 'warned %s\n' "$class"
            return 0
            ;;
        real_file)
            printf 'warned %s\n' "$class"
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Aggregate progress parsing  (W4 - issue e684610)
# ---------------------------------------------------------------------------

# rows_rollup_from_listing  (newline list of "<id> <state>" on stdin)
# state in {done,failed,running,stale_started,pending}. Prints a single
# machine-readable roll-up including stale reservations.
rows_rollup_from_listing() {
    local id state done=0 failed=0 running=0 stale=0 pending=0 total=0
    while read -r id state; do
        [ -z "$id" ] && continue
        total=$((total + 1))
        case "$state" in
            done) done=$((done + 1)) ;;
            failed) failed=$((failed + 1)) ;;
            running) running=$((running + 1)) ;;
            stale_started) stale=$((stale + 1)) ;;
            *) pending=$((pending + 1)) ;;
        esac
    done
    printf 'rows_done=%s rows_failed=%s rows_running=%s rows_stale=%s rows_pending=%s rows_total=%s\n' \
        "$done" "$failed" "$running" "$stale" "$pending" "$total"
}

# max_checkpoint_step <dir>  -> highest integer checkpoint step found under dir.
# Looks for files/dirs whose names contain a step integer (e.g. checkpoint_1200,
# step-1200, 1200.eqx). Prints the max, or "none" if none found. Local FS only;
# the driver runs the remote equivalent over SSH.
max_checkpoint_step() {
    local dir=$1 max="none" name n
    [ -d "$dir" ] || { printf 'none\n'; return 0; }
    while IFS= read -r name; do
        # Extract the last run of digits in the basename.
        n=$(printf '%s' "${name##*/}" | grep -oE '[0-9]+' | tail -1 || true)
        [ -z "$n" ] && continue
        if [ "$max" = "none" ] || [ "$n" -gt "$max" ]; then
            max=$n
        fi
    done < <(find "$dir" -maxdepth 2 \( -type f -o -type d \) 2>/dev/null)
    printf '%s\n' "$max"
}

# discover_row_ids <sentinel_dir>  -> one row id per line.
# Row identity is derived from the per-row marker files the launcher writes
# (<id>.started / <id>.pid / <id>.done / <id>.failed) so no jq / manifest is
# needed remotely. The synchronous `.started` reservation marker (FIX 3) makes a
# row discoverable the instant it is launched, before the backgrounded job has
# written its `.pid`. Falls back to nothing when the dir is absent.
discover_row_ids() {
    local sentinel_dir=$1 f base
    [ -d "$sentinel_dir" ] || return 0
    for f in "$sentinel_dir"/*.started "$sentinel_dir"/*.pid \
             "$sentinel_dir"/*.done "$sentinel_dir"/*.failed; do
        [ -e "$f" ] || continue
        base=${f##*/}
        base=${base%.started}; base=${base%.pid}
        base=${base%.done}; base=${base%.failed}
        printf '%s\n' "$base"
    done | sort -u
}

# row_state <sentinel_dir> <id>  -> done|failed|running|stale_started|pending.
#   <id>.done present              -> done
#   <id>.failed present            -> failed
#   <id>.pid is live               -> running
#   <id>.started or stale <id>.pid -> stale_started
#   otherwise                      -> pending
row_state() {
    local sentinel_dir=$1 id=$2 pid
    if [ -f "$sentinel_dir/$id.done" ]; then printf 'done\n'
    elif [ -f "$sentinel_dir/$id.failed" ]; then printf 'failed\n'
    elif [ -f "$sentinel_dir/$id.pid" ]; then
        pid=$(cat "$sentinel_dir/$id.pid" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            printf 'running\n'
        else
            printf 'stale_started\n'
        fi
    elif [ -f "$sentinel_dir/$id.started" ]; then printf 'stale_started\n'
    else printf 'pending\n'; fi
}

# progress_report <sentinel_dir> <checkpoint_dir> <log_dir>
# Emits ONE machine-readable status line (no jq needed by the consumer):
#   rows_done=k rows_failed=j rows_running=r rows_stale=s rows_pending=p rows_total=N \
#   last_batch=<n|none> last_checkpoint=<step|none> rows=<id:state,...>
# Distinguishes "compiling" (running, last_batch=none, last_checkpoint=none)
# from "training" (a batch/checkpoint advancing) from "stalled" (running but no
# recent progress — the caller compares last_batch/last_checkpoint across polls).
progress_report() {
    local sentinel_dir=$1 checkpoint_dir=$2 log_dir=$3
    local id state listing="" detail="" ckpt batch
    while IFS= read -r id; do
        [ -z "$id" ] && continue
        state=$(row_state "$sentinel_dir" "$id")
        listing="${listing}${id} ${state}"$'\n'
        detail="${detail}${detail:+,}${id}:${state}"
    done < <(discover_row_ids "$sentinel_dir")

    local rollup
    if [ -n "$listing" ]; then
        rollup=$(printf '%s' "$listing" | rows_rollup_from_listing)
    else
        rollup="rows_done=0 rows_failed=0 rows_running=0 rows_stale=0 rows_pending=0 rows_total=0"
    fi

    ckpt=$(max_checkpoint_step "$checkpoint_dir")
    # Latest batch counter across all per-row logs.
    batch="none"
    if [ -d "$log_dir" ]; then
        local lf b
        for lf in "$log_dir"/*.log; do
            [ -e "$lf" ] || continue
            b=$(last_batch_from_log "$lf")
            [ "$b" = "none" ] && continue
            if [ "$batch" = "none" ] || [ "$b" -gt "$batch" ]; then batch=$b; fi
        done
    fi

    printf '%s last_batch=%s last_checkpoint=%s rows=%s\n' \
        "$rollup" "$batch" "$ckpt" "${detail:-none}"
}

# last_batch_from_log <logfile>  -> last batch/step counter grepped from a log.
# Matches common patterns: "batch 1234", "step=1234", "step 1234", "it 1234".
# Prints the last numeric match, or "none".
last_batch_from_log() {
    local log=$1 val
    [ -f "$log" ] || { printf 'none\n'; return 0; }
    val=$(grep -oiE '(batch|step|iter|it)[[:space:]=:]+[0-9]+' "$log" 2>/dev/null |
        grep -oE '[0-9]+' | tail -1 || true)
    printf '%s\n' "${val:-none}"
}
