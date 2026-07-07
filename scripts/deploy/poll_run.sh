#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib_acquire.sh
source "$SCRIPT_DIR/lib_acquire.sh"

EARLY_CADENCE_SECONDS="${EARLY_CADENCE_SECONDS:-300}"
STEADY_CADENCE_SECONDS="${STEADY_CADENCE_SECONDS:-1800}"
EARLY_WINDOW_SECONDS="${EARLY_WINDOW_SECONDS:-3600}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-~/.runpod/ssh/RunPod-Key-Go}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-}"
REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-}"
REMOTE_RLRMP_ROOT="${REMOTE_RLRMP_ROOT:-/workspace/rlrmp}"
# Where to look for checkpoint step dirs and per-row training logs on the pod.
REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-}"
REMOTE_DEPLOY_CONFIG_PATH="${REMOTE_DEPLOY_CONFIG_PATH:-/workspace/feedbax_runs/runpod-deploy-config.json}"

POD_ID="${POD_ID:-}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"
ENDPOINT_SOURCE="missing"
ENDPOINT_CLASSIFICATION="missing_direct_endpoint"
SSH_ERROR="missing_ssh_metadata"
STARTED_AT_EPOCH="${STARTED_AT_EPOCH:-}"
CADENCE_SECONDS=""
RUN_CONFIG=""
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/deploy/poll_run.sh --pod-id <id> [options]

Options:
  --pod-id <id>              RunPod pod id to inspect.
  --ssh-host <host>          Known SSH host; otherwise read from runpodctl pod get.
  --ssh-port <port>          Known SSH port; otherwise read from runpodctl pod get.
  --run-config <json>        Local run config written by runpod_deploy.sh.
  --remote-run-dir <path>    Explicit remote run directory when no run config is used.
  --deploy-config-path <p>   Remote deploy config path for run-dir derivation.
  --started-at-epoch <sec>   Used to choose early vs steady cadence.
  --cadence-seconds <sec>    Override sleep cadence for this call.
  --dry-run                  Print commands without executing; does not sleep.
  -h, --help                 Show this help.

The script prints exactly one status line, then sleeps for the selected cadence.
USAGE
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

print_cmd() {
    local arg
    printf '+'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

capture_cmd() {
    print_cmd "$@" >&2
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

expand_path() {
    local value=$1
    if [[ $value == "~"* ]]; then
        printf '%s%s\n' "$HOME" "${value:1}"
    else
        printf '%s\n' "$value"
    fi
}

safe_status_value() {
    local value=${1:-none}
    value=${value:-none}
    printf '%s\n' "$value" | tr '[:space:]' '_' | tr -cd '[:alnum:]_.:/=@,+-'
}

remote_capture() {
    local command=$1 key_path
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    ssh -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
        -i "$key_path" \
        -p "$SSH_PORT" \
        "root@$SSH_HOST" \
        "$command"
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --pod-id)
                shift
                [ "$#" -gt 0 ] || die "--pod-id requires a value"
                POD_ID=$1
                ;;
            --ssh-host)
                shift
                [ "$#" -gt 0 ] || die "--ssh-host requires a value"
                SSH_HOST=$1
                ;;
            --ssh-port)
                shift
                [ "$#" -gt 0 ] || die "--ssh-port requires a value"
                SSH_PORT=$1
                ;;
            --run-config)
                shift
                [ "$#" -gt 0 ] || die "--run-config requires a value"
                RUN_CONFIG=$1
                ;;
            --remote-run-dir)
                shift
                [ "$#" -gt 0 ] || die "--remote-run-dir requires a value"
                REMOTE_RUN_DIR=$1
                ;;
            --deploy-config-path)
                shift
                [ "$#" -gt 0 ] || die "--deploy-config-path requires a value"
                REMOTE_DEPLOY_CONFIG_PATH=$1
                ;;
            --started-at-epoch)
                shift
                [ "$#" -gt 0 ] || die "--started-at-epoch requires a value"
                STARTED_AT_EPOCH=$1
                ;;
            --cadence-seconds)
                shift
                [ "$#" -gt 0 ] || die "--cadence-seconds requires a value"
                CADENCE_SECONDS=$1
                ;;
            --dry-run)
                DRY_RUN=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown option: $1"
                ;;
        esac
        shift
    done
}

derive_remote_paths() {
    local config=""
    if [ -n "$RUN_CONFIG" ]; then
        [ -f "$RUN_CONFIG" ] || die "run config not found: $RUN_CONFIG"
        config=$(cat "$RUN_CONFIG")
    elif [ -z "$REMOTE_RUN_DIR" ]; then
        [ "$DRY_RUN" -eq 0 ] ||
            die "remote run dir is required in --dry-run; pass --run-config or --remote-run-dir"
        [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] ||
            die "remote run dir is required; pass --run-config or --remote-run-dir"
        config=$(remote_capture "test -f $(sq "$REMOTE_DEPLOY_CONFIG_PATH") && cat $(sq "$REMOTE_DEPLOY_CONFIG_PATH")") ||
            die "remote run dir is not set and deploy config was not found at $REMOTE_DEPLOY_CONFIG_PATH"
    fi
    if [ -n "$config" ]; then
        REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-$(printf '%s' "$config" | jq -r '.remote_run_dir // empty')}"
        REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-$(printf '%s' "$config" | jq -r '.remote_sentinel_dir // empty')}"
        REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-$(printf '%s' "$config" | jq -r '.remote_checkpoint_dir // empty')}"
        REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$(printf '%s' "$config" | jq -r '.remote_log_dir // empty')}"
    fi
    [ -n "$REMOTE_RUN_DIR" ] ||
        die "remote run dir is required; pass --run-config or --remote-run-dir"
    REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-$REMOTE_RUN_DIR/sentinels}"
    REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-$REMOTE_RUN_DIR}"
    REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_RUN_DIR/logs}"
}

choose_cadence() {
    if [ -n "$CADENCE_SECONDS" ]; then
        printf '%s\n' "$CADENCE_SECONDS"
        return 0
    fi
    if [ -n "$STARTED_AT_EPOCH" ]; then
        local now elapsed
        now=$(date +%s)
        elapsed=$((now - STARTED_AT_EPOCH))
        if [ "$elapsed" -ge "$EARLY_WINDOW_SECONDS" ]; then
            printf '%s\n' "$STEADY_CADENCE_SECONDS"
            return 0
        fi
    fi
    printf '%s\n' "$EARLY_CADENCE_SECONDS"
}

extract_ssh() {
    local detail=$1
    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        ENDPOINT_SOURCE="provided"
        ENDPOINT_CLASSIFICATION="known_direct_endpoint"
        SSH_ERROR="none"
        return 0
    fi

    SSH_HOST=$(printf '%s\n' "$detail" | jq -r '(.ssh.ip // .ssh.host // empty)')
    SSH_PORT=$(printf '%s\n' "$detail" | jq -r '(.ssh.port // empty)')
    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        ENDPOINT_SOURCE="ssh_object"
        ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
        SSH_ERROR="none"
        return 0
    fi

    if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
        local ssh_command
        ssh_command=$(printf '%s\n' "$detail" |
            jq -r '(.ssh.sshCommand // .ssh.ssh_command // .ssh.command // empty)')
        if [ -z "$SSH_PORT" ] && [[ $ssh_command =~ -p[[:space:]]+([0-9]+) ]]; then
            SSH_PORT=${BASH_REMATCH[1]}
        fi
        if [ -z "$SSH_HOST" ] && [[ $ssh_command =~ root@([^[:space:]]+) ]]; then
            SSH_HOST=${BASH_REMATCH[1]}
        fi
    fi
    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        ENDPOINT_SOURCE="ssh_command"
        ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
        SSH_ERROR="none"
        return 0
    fi

    local ssh_error
    ssh_error=$(printf '%s\n' "$detail" |
        jq -r '
            .ssh as $ssh
            | (
                if ($ssh | type) == "object" then
                    ($ssh.error // $ssh.message // $ssh.reason // empty)
                else
                    empty
                end
              ) // .sshError // .runtime.sshError // empty
        ')
    SSH_ERROR=$(safe_status_value "${ssh_error:-missing_ssh_metadata}")
    if [ -n "$SSH_HOST" ] || [ -n "$SSH_PORT" ]; then
        ENDPOINT_SOURCE="partial"
        ENDPOINT_CLASSIFICATION="partial_direct_endpoint"
    else
        ENDPOINT_SOURCE="missing"
        ENDPOINT_CLASSIFICATION="missing_direct_endpoint"
    fi
}

# build_remote_status_command renders the self-contained remote bash that the
# pod runs over SSH. It reports gpu, bootstrap sentinels, and the per-row
# aggregate progress (rows roll-up, last_checkpoint, last_batch) so "compiling"
# is distinguishable from "training" from "stalled". The progress block mirrors
# lib_acquire.sh::progress_report, which is unit-tested locally over a mock
# remote layout. One status line, no jq.
build_remote_status_command() {
    local sentinel_dir=$1 checkpoint_dir=$2 log_dir=$3
    cat <<REMOTE
gpu=\$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
gpu_safe=\$(printf '%s' "\${gpu:-unknown}" | tr ' ' '_')
printf 'ssh=ready gpu=%s ' "\$gpu_safe"
for name in uv_sync jax_cuda venv_probe probe_ok probe_failed_rebuilding rebuild_done; do
  if [ -f '$sentinel_dir/'"\$name"'.done' ]; then printf '%s=done ' "\$name"
  elif [ -f '$sentinel_dir/'"\$name"'.failed' ]; then printf '%s=failed ' "\$name"
  else printf '%s=pending ' "\$name"; fi
done
sdir='$sentinel_dir'
ids=\$( { for f in "\$sdir"/*.started "\$sdir"/*.pid "\$sdir"/*.done "\$sdir"/*.failed; do
           [ -e "\$f" ] || continue
           b=\${f##*/}; b=\${b%.started}; b=\${b%.pid}; b=\${b%.done}; b=\${b%.failed}
           if [[ "\$b" == uv_sync || "\$b" == jax_cuda || "\$b" == venv_probe || "\$b" == probe_ok || "\$b" == probe_failed_rebuilding || "\$b" == rebuild_done ]]; then
             continue
           fi
           echo "\$b"
         done; } | sort -u )
nd=0; nf=0; nr=0; ns=0; np=0; nt=0; detail=''
for id in \$ids; do
  nt=\$((nt+1))
  if [ -f "\$sdir/\$id.done" ]; then st=done; nd=\$((nd+1))
  elif [ -f "\$sdir/\$id.failed" ]; then st=failed; nf=\$((nf+1))
  elif [ -f "\$sdir/\$id.pid" ]; then
    pid=\$(cat "\$sdir/\$id.pid" 2>/dev/null || true)
    if [ -n "\$pid" ] && kill -0 "\$pid" 2>/dev/null; then st=running; nr=\$((nr+1))
    else st=stale_started; ns=\$((ns+1)); fi
  elif [ -f "\$sdir/\$id.started" ]; then st=stale_started; ns=\$((ns+1))
  else st=pending; np=\$((np+1)); fi
  detail="\${detail:+\$detail,}\$id:\$st"
done
printf 'rows_done=%s rows_failed=%s rows_running=%s rows_stale=%s rows_pending=%s rows_total=%s ' "\$nd" "\$nf" "\$nr" "\$ns" "\$np" "\$nt"
train_process_state=absent
if pgrep -af '[t]rain_|[p]ython .*train|[u]v run .*train' >/dev/null 2>&1; then
  train_process_state=present
fi
ckpt=none
for d in \$(find '$checkpoint_dir' -maxdepth 2 \( -type f -o -type d \) 2>/dev/null); do
  n=\$(printf '%s' "\${d##*/}" | grep -oE '[0-9]+' | tail -1)
  [ -z "\$n" ] && continue
  if [ "\$ckpt" = none ] || [ "\$n" -gt "\$ckpt" ]; then ckpt=\$n; fi
done
batch=none
for lf in '$log_dir'/*.log; do
  [ -e "\$lf" ] || continue
  b=\$(grep -oiE '(batch|step|iter|it)[[:space:]=:]+[0-9]+' "\$lf" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  [ -z "\$b" ] && continue
  if [ "\$batch" = none ] || [ "\$b" -gt "\$batch" ]; then batch=\$b; fi
done
printf 'last_checkpoint=%s last_batch=%s train_process=%s rows=%s' "\$ckpt" "\$batch" "\$train_process_state" "\${detail:-none}"
REMOTE
}

remote_status() {
    if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
        SSH_ERROR=${SSH_ERROR:-missing_ssh_endpoint}
        printf 'ssh=missing gpu=unknown uv_sync=unknown jax_cuda=unknown venv_probe=unknown probe_ok=unknown probe_failed_rebuilding=unknown rebuild_done=unknown rows_done=0 rows_failed=0 rows_running=0 rows_stale=0 rows_pending=0 rows_total=0 last_checkpoint=none last_batch=none rows=none'
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        print_cmd ssh -p "$SSH_PORT" "root@$SSH_HOST" \
            "nvidia-smi && test -d '$REMOTE_SENTINEL_DIR'" >&2
        printf 'ssh=dry-run gpu=dry-run uv_sync=dry-run jax_cuda=dry-run venv_probe=dry-run probe_ok=dry-run probe_failed_rebuilding=dry-run rebuild_done=dry-run rows_done=0 rows_failed=0 rows_running=0 rows_stale=0 rows_pending=0 rows_total=0 last_checkpoint=none last_batch=none rows=none'
        return 0
    fi

    local key_path output remote_cmd
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    remote_cmd=$(build_remote_status_command \
        "$REMOTE_SENTINEL_DIR" "$REMOTE_CHECKPOINT_DIR" "$REMOTE_LOG_DIR")
    if ! output=$(ssh -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
        -i "$key_path" \
        -p "$SSH_PORT" \
        "root@$SSH_HOST" \
        "$remote_cmd"); then
        SSH_ERROR="ssh_probe_failed"
        printf 'ssh=failed gpu=unknown uv_sync=unknown jax_cuda=unknown venv_probe=unknown probe_ok=unknown probe_failed_rebuilding=unknown rebuild_done=unknown rows_done=0 rows_failed=0 rows_running=0 rows_stale=0 rows_pending=0 rows_total=0 last_checkpoint=none last_batch=none rows=none'
        return 0
    fi
    SSH_ERROR="none"
    printf '%s' "$output"
}

main() {
    parse_args "$@"
    command -v jq >/dev/null 2>&1 || die "required command not found: jq"
    [ -n "$POD_ID" ] || die "--pod-id is required"

    local cadence detail desired_status timestamp status endpoint_context
    cadence=$(choose_cadence)
    if [ "$DRY_RUN" -eq 1 ]; then
        detail='{"desiredStatus":"DRY_RUN","ssh":{"ip":"dry-run-host","port":22}}'
    else
        detail=$(capture_cmd runpodctl pod get "$POD_ID" --output json)
    fi

    desired_status=$(printf '%s\n' "$detail" |
        jq -r '(.desiredStatus // .status // .runtime.status // "unknown")')
    extract_ssh "$detail"
    derive_remote_paths
    status=$(remote_status)
    endpoint_context="endpoint_source=$ENDPOINT_SOURCE endpoint_classification=$ENDPOINT_CLASSIFICATION ssh_error=${SSH_ERROR:-none}"
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    printf '%s pod=%s pod_status=%s %s %s\n' \
        "$timestamp" "$POD_ID" "$desired_status" "$endpoint_context" "$status"
    if [ "$DRY_RUN" -eq 1 ]; then
        run_cmd sleep "$cadence"
    else
        sleep "$cadence"
    fi
}

main "$@"
