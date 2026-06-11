#!/usr/bin/env bash
set -euo pipefail

EARLY_CADENCE_SECONDS="${EARLY_CADENCE_SECONDS:-300}"
STEADY_CADENCE_SECONDS="${STEADY_CADENCE_SECONDS:-1800}"
EARLY_WINDOW_SECONDS="${EARLY_WINDOW_SECONDS:-3600}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-~/.runpod/ssh/RunPod-Key-Go}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/workspace/feedbax_runs/runpod-deploy}"
REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-$REMOTE_RUN_DIR/sentinels}"

POD_ID="${POD_ID:-}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"
STARTED_AT_EPOCH="${STARTED_AT_EPOCH:-}"
CADENCE_SECONDS=""
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/deploy/poll_run.sh --pod-id <id> [options]

Options:
  --pod-id <id>              RunPod pod id to inspect.
  --ssh-host <host>          Known SSH host; otherwise read from runpodctl pod get.
  --ssh-port <port>          Known SSH port; otherwise read from runpodctl pod get.
  --started-at-epoch <sec>   Used to choose early vs steady cadence.
  --cadence-seconds <sec>    Override sleep cadence for this call.
  --dry-run                  Print commands without executing; does not sleep.
  -h, --help                 Show this help.

The script sleeps internally, then prints exactly one status line.
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
    [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && return 0
    SSH_HOST=$(printf '%s\n' "$detail" | jq -r '(.ssh.ip // .ssh.host // empty)')
    SSH_PORT=$(printf '%s\n' "$detail" | jq -r '(.ssh.port // empty)')
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
}

remote_status() {
    if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
        printf 'ssh=missing gpu=unknown uv=unknown jax=unknown training=unknown done=unknown failed=unknown'
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        print_cmd ssh -p "$SSH_PORT" "root@$SSH_HOST" \
            "nvidia-smi && test -d '$REMOTE_SENTINEL_DIR'" >&2
        printf 'ssh=dry-run gpu=dry-run uv=dry-run jax=dry-run training=dry-run done=dry-run failed=dry-run'
        return 0
    fi

    local key_path output
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    if ! output=$(ssh -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout="$SSH_CONNECT_TIMEOUT" \
        -i "$key_path" \
        -p "$SSH_PORT" \
        "root@$SSH_HOST" \
        "gpu=\$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true); \
         gpu_safe=\$(printf '%s' \"\${gpu:-unknown}\" | tr ' ' '_'); \
         printf 'ssh=ready gpu=%s ' \"\$gpu_safe\"; \
         for name in uv_sync jax_cuda training; do \
           if [ -f '$REMOTE_SENTINEL_DIR/'\"\$name\"'.done' ]; then printf '%s=done ' \"\$name\"; \
           elif [ -f '$REMOTE_SENTINEL_DIR/'\"\$name\"'.failed' ]; then printf '%s=failed ' \"\$name\"; \
           else printf '%s=pending ' \"\$name\"; fi; \
         done; \
         if [ -f '$REMOTE_SENTINEL_DIR/training.done' ]; then printf 'done=true '; else printf 'done=false '; fi; \
         if ls '$REMOTE_SENTINEL_DIR/'*.failed >/dev/null 2>&1; then printf 'failed=true'; else printf 'failed=false'; fi"); then
        printf 'ssh=failed gpu=unknown uv=unknown jax=unknown training=unknown done=unknown failed=unknown'
        return 0
    fi
    printf '%s' "$output"
}

main() {
    parse_args "$@"
    [ -n "$POD_ID" ] || die "--pod-id is required"
    command -v jq >/dev/null 2>&1 || die "required command not found: jq"

    local cadence detail desired_status timestamp status
    cadence=$(choose_cadence)
    if [ "$DRY_RUN" -eq 1 ]; then
        run_cmd sleep "$cadence"
        detail='{"desiredStatus":"DRY_RUN","ssh":{"ip":"dry-run-host","port":22}}'
    else
        sleep "$cadence"
        detail=$(capture_cmd runpodctl pod get "$POD_ID" --output json)
    fi

    desired_status=$(printf '%s\n' "$detail" |
        jq -r '(.desiredStatus // .status // .runtime.status // "unknown")')
    extract_ssh "$detail"
    status=$(remote_status)
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    printf '%s pod=%s pod_status=%s %s\n' "$timestamp" "$POD_ID" "$desired_status" "$status"
}

main "$@"
