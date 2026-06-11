#!/usr/bin/env bash
set -euo pipefail

# Deterministic RunPod deploy for the rlrmp/feedbax/jax-cookbook workflow.
# Override these values with environment variables or `--config <file>`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
FEEDBAX_ROOT="${FEEDBAX_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
PROJECTS_ROOT="${PROJECTS_ROOT:-$(cd "$FEEDBAX_ROOT/../.." && pwd -P)}"
UTILS_ROOT="${UTILS_ROOT:-$(cd "$FEEDBAX_ROOT/../../.." && pwd -P)/05 Utils}"
RLRMP_ROOT="${RLRMP_ROOT:-$PROJECTS_ROOT/rlrmp}"
JAX_COOKBOOK_ROOT="${JAX_COOKBOOK_ROOT:-$UTILS_ROOT/jax-cookbook}"

RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204}"
RUNPOD_GPU_ID="${RUNPOD_GPU_ID:-NVIDIA GeForce RTX 5090}"
RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-30}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-30}"
RUNPOD_VOLUME_MOUNT="${RUNPOD_VOLUME_MOUNT:-/workspace}"
RUNPOD_PORTS="${RUNPOD_PORTS:-22/tcp,8080/http}"
RUNPOD_NAME="${RUNPOD_NAME:-feedbax-rlrmp-$(date +%Y%m%d-%H%M%S)}"
RUNPOD_DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-}"
RUNPOD_ENV_JSON="${RUNPOD_ENV_JSON:-{}}"

RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-~/.runpod/ssh/RunPod-Key-Go}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
READINESS_TIMEOUT_SECONDS="${READINESS_TIMEOUT_SECONDS:-900}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-15}"
SENTINEL_TIMEOUT_SECONDS="${SENTINEL_TIMEOUT_SECONDS:-7200}"
SENTINEL_POLL_SECONDS="${SENTINEL_POLL_SECONDS:-30}"

REMOTE_RLRMP_ROOT="${REMOTE_RLRMP_ROOT:-}"
REMOTE_FEEDBAX_ROOT="${REMOTE_FEEDBAX_ROOT:-}"
REMOTE_JAX_COOKBOOK_ROOT="${REMOTE_JAX_COOKBOOK_ROOT:-}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-}"
REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-}"
TRAIN_COMMAND="${TRAIN_COMMAND:-}"

CONFIG_FILE=""
DRY_RUN=0
POD_ID="${POD_ID:-}"
TRAIN_SPEC="${TRAIN_SPEC:-}"
SKIP_IMAGE_CHECK=0

SSH_HOST=""
SSH_PORT=""

usage() {
    cat <<'USAGE'
Usage: scripts/deploy/runpod_deploy.sh [options]

Options:
  --config <file>           Source a bash config file after defaults.
  --dry-run                 Print commands without executing them.
  --pod-id <id>             Reuse an existing RunPod pod instead of creating one.
  --train-spec <json>       Required when a training launch command is configured.
  --launch-command <cmd>    Remote training command to launch after bootstrap.
  --skip-image-check        Skip Docker Hub tag existence check.
  -h, --help                Show this help.

The training launch is refused unless --train-spec points to JSON containing
`"user_confirmed": true`; failure prints the spec table and exits 2.
USAGE
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '==> %s\n' "$*" >&2
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

sq() {
    local value=${1-}
    value=${value//\'/\'\\\'\'}
    printf "'%s'" "$value"
}

expand_path() {
    local value=$1
    if [[ $value == "~"* ]]; then
        printf '%s%s\n' "$HOME" "${value:1}"
    else
        printf '%s\n' "$value"
    fi
}

first_pass_config() {
    local args=("$@")
    local i=0
    while [ "$i" -lt "${#args[@]}" ]; do
        case "${args[$i]}" in
            --config)
                [ "$((i + 1))" -lt "${#args[@]}" ] || die "--config requires a file"
                CONFIG_FILE=${args[$((i + 1))]}
                i=$((i + 2))
                ;;
            *)
                i=$((i + 1))
                ;;
        esac
    done
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --config)
                shift
                [ "$#" -gt 0 ] || die "--config requires a file"
                CONFIG_FILE=$1
                ;;
            --dry-run)
                DRY_RUN=1
                ;;
            --pod-id)
                shift
                [ "$#" -gt 0 ] || die "--pod-id requires a value"
                POD_ID=$1
                ;;
            --train-spec)
                shift
                [ "$#" -gt 0 ] || die "--train-spec requires a JSON file"
                TRAIN_SPEC=$1
                ;;
            --launch-command)
                shift
                [ "$#" -gt 0 ] || die "--launch-command requires a command string"
                TRAIN_COMMAND=$1
                ;;
            --skip-image-check)
                SKIP_IMAGE_CHECK=1
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

load_config() {
    [ -z "$CONFIG_FILE" ] && return 0
    [ -f "$CONFIG_FILE" ] || die "config file not found: $CONFIG_FILE"
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        die "required command not found: $1"
    fi
}

require_real_commands() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    require_command curl
    require_command jq
    require_command perl
    require_command rsync
    require_command runpodctl
    require_command ssh
}

default_path_patches() {
    if [ -n "${PATH_PATCHES:-}" ]; then
        return 0
    fi
    PATH_PATCHES="$(printf '%s\n%s\n' \
        "$REMOTE_RLRMP_ROOT/pyproject.toml|$FEEDBAX_ROOT|$REMOTE_FEEDBAX_ROOT" \
        "$REMOTE_FEEDBAX_ROOT/pyproject.toml|$JAX_COOKBOOK_ROOT|$REMOTE_JAX_COOKBOOK_ROOT")"
}

finalize_remote_defaults() {
    REMOTE_RLRMP_ROOT="${REMOTE_RLRMP_ROOT:-$RUNPOD_VOLUME_MOUNT/rlrmp}"
    REMOTE_FEEDBAX_ROOT="${REMOTE_FEEDBAX_ROOT:-$RUNPOD_VOLUME_MOUNT/feedbax}"
    REMOTE_JAX_COOKBOOK_ROOT="${REMOTE_JAX_COOKBOOK_ROOT:-$RUNPOD_VOLUME_MOUNT/jax-cookbook}"
    REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-$RUNPOD_VOLUME_MOUNT/feedbax_runs/runpod-deploy}"
    REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-$REMOTE_RUN_DIR/sentinels}"
}

print_spec_table() {
    local spec=${1-}
    printf 'Training spec confirmation required.\n' >&2
    printf '%-28s %s\n' "field" "value" >&2
    printf '%-28s %s\n' "-----" "-----" >&2
    if [ -z "$spec" ]; then
        printf '%-28s %s\n' "train_spec" "<missing>" >&2
        return 0
    fi
    if [ ! -f "$spec" ]; then
        printf '%-28s %s\n' "train_spec" "$spec (not found)" >&2
        return 0
    fi
    if ! jq empty "$spec" >/dev/null 2>&1; then
        printf '%-28s %s\n' "train_spec" "$spec (invalid JSON)" >&2
        return 0
    fi
    jq -r '
        to_entries[]
        | [
            .key,
            (.value | if type == "object" or type == "array" then tojson else tostring end)
          ]
        | @tsv
    ' "$spec" |
        while IFS="$(printf '\t')" read -r key value; do
            printf '%-28s %s\n' "$key" "$value" >&2
        done
}

validate_train_spec_gate() {
    [ -n "$TRAIN_COMMAND" ] || return 0
    if [ -z "$TRAIN_SPEC" ] || [ ! -f "$TRAIN_SPEC" ]; then
        print_spec_table "$TRAIN_SPEC"
        exit 2
    fi
    if ! jq -e '.user_confirmed == true' "$TRAIN_SPEC" >/dev/null; then
        print_spec_table "$TRAIN_SPEC"
        exit 2
    fi
}

parse_docker_hub_image() {
    local image=$1
    local name tag
    image=${image#docker.io/}
    image=${image%%@*}
    if [[ $image == *:* ]]; then
        tag=${image##*:}
        name=${image%:*}
    else
        tag=latest
        name=$image
    fi
    if [[ $name == */* ]]; then
        DOCKER_HUB_REPO=$name
    else
        DOCKER_HUB_REPO="library/$name"
    fi
    DOCKER_HUB_TAG=$tag
}

check_docker_hub_tag() {
    [ "$SKIP_IMAGE_CHECK" -eq 1 ] && return 0
    local url
    parse_docker_hub_image "$RUNPOD_IMAGE"
    url="https://hub.docker.com/v2/repositories/$DOCKER_HUB_REPO/tags/$DOCKER_HUB_TAG"
    run_cmd curl --fail --silent --show-error --location "$url" --output /dev/null
}

create_pod() {
    if [ -n "$POD_ID" ]; then
        log "using existing pod $POD_ID"
        return 0
    fi

    local cmd output pod_id
    cmd=(
        runpodctl pod create
        --image "$RUNPOD_IMAGE"
        --gpu-id "$RUNPOD_GPU_ID"
        --gpu-count "$RUNPOD_GPU_COUNT"
        --cloud-type "$RUNPOD_CLOUD_TYPE"
        --container-disk-in-gb "$RUNPOD_CONTAINER_DISK_GB"
        --volume-in-gb "$RUNPOD_VOLUME_GB"
        --volume-mount-path "$RUNPOD_VOLUME_MOUNT"
        --ports "$RUNPOD_PORTS"
        --name "$RUNPOD_NAME"
        --env "$RUNPOD_ENV_JSON"
        --output json
    )
    if [ -n "$RUNPOD_DATA_CENTER_IDS" ]; then
        cmd+=(--data-center-ids "$RUNPOD_DATA_CENTER_IDS")
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        run_cmd "${cmd[@]}"
        POD_ID="dry-run-pod"
        return 0
    fi

    output=$(capture_cmd "${cmd[@]}")
    pod_id=$(printf '%s\n' "$output" |
        jq -r '(.id // .pod.id // .podId // .data.id // .data.pod.id // empty)')
    [ -n "$pod_id" ] || die "could not parse pod id from runpodctl output"
    POD_ID=$pod_id
    log "created pod $POD_ID"
}

extract_ssh_field() {
    local field=$1
    jq -r --arg field "$field" '
        .ssh as $ssh
        | if ($ssh | type) == "object" then
            ($ssh[$field] // empty)
          else
            empty
          end
    '
}

extract_ssh_command() {
    jq -r '
        .ssh as $ssh
        | if ($ssh | type) == "object" then
            ($ssh.sshCommand // $ssh.ssh_command // $ssh.command // empty)
          else
            empty
          end
    '
}

parse_ssh_command_fields() {
    local ssh_command=$1
    if [ -z "$SSH_PORT" ] && [[ $ssh_command =~ -p[[:space:]]+([0-9]+) ]]; then
        SSH_PORT=${BASH_REMATCH[1]}
    fi
    if [ -z "$SSH_HOST" ] && [[ $ssh_command =~ root@([^[:space:]]+) ]]; then
        SSH_HOST=${BASH_REMATCH[1]}
    fi
}

remote_cmd() {
    local command=$1
    local key_path
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    local cmd=(ssh
        -o BatchMode=yes
        -o StrictHostKeyChecking=accept-new
        -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    )
    if [ -n "$key_path" ]; then
        cmd+=(-i "$key_path")
    fi
    cmd+=(-p "$SSH_PORT" "root@$SSH_HOST" "$command")
    run_cmd "${cmd[@]}"
}

remote_capture() {
    local command=$1
    local key_path
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    local cmd=(ssh
        -o BatchMode=yes
        -o StrictHostKeyChecking=accept-new
        -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    )
    if [ -n "$key_path" ]; then
        cmd+=(-i "$key_path")
    fi
    cmd+=(-p "$SSH_PORT" "root@$SSH_HOST" "$command")
    capture_cmd "${cmd[@]}"
}

probe_ssh_gpu() {
    remote_cmd "nvidia-smi >/dev/null"
}

wait_for_ssh_ready() {
    if [ "$DRY_RUN" -eq 1 ]; then
        run_cmd runpodctl pod get "$POD_ID" --output json
        SSH_HOST="dry-run-host"
        SSH_PORT="22"
        probe_ssh_gpu
        return 0
    fi

    local deadline detail ssh_ip ssh_port ssh_command
    deadline=$((SECONDS + READINESS_TIMEOUT_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        detail=$(capture_cmd runpodctl pod get "$POD_ID" --output json)
        ssh_ip=$(printf '%s\n' "$detail" | extract_ssh_field ip)
        ssh_port=$(printf '%s\n' "$detail" | extract_ssh_field port)
        if [ -z "$ssh_ip" ]; then
            ssh_ip=$(printf '%s\n' "$detail" | extract_ssh_field host)
        fi
        SSH_HOST=$ssh_ip
        SSH_PORT=$ssh_port
        if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
            ssh_command=$(printf '%s\n' "$detail" | extract_ssh_command)
            parse_ssh_command_fields "$ssh_command"
        fi
        if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && probe_ssh_gpu; then
            log "pod ssh and nvidia-smi are ready"
            return 0
        fi
        sleep "$READINESS_POLL_SECONDS"
    done
    die "timed out waiting for .ssh object and functional nvidia-smi probe"
}

rsync_rsh() {
    local key_path quoted_key rsh
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    rsh="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -p $SSH_PORT"
    if [ -n "$key_path" ]; then
        printf -v quoted_key '%q' "$key_path"
        rsh="$rsh -i $quoted_key"
    fi
    printf '%s\n' "$rsh"
}

rsync_repo() {
    local name=$1
    local source=$2
    local target=$3
    local rsh
    [ -d "$source" ] || die "$name source directory not found: $source"
    rsh=$(rsync_rsh)
    run_cmd rsync -az --delete --no-owner --no-group --stats \
        --exclude .git \
        --exclude .venv \
        --exclude __pycache__ \
        --exclude .pytest_cache \
        --exclude .mypy_cache \
        --exclude .ruff_cache \
        --exclude .DS_Store \
        --exclude node_modules \
        --exclude web/node_modules \
        --exclude web/dist \
        -e "$rsh" \
        "$source/" "root@$SSH_HOST:$target/"
}

sync_repos() {
    rsync_repo rlrmp "$RLRMP_ROOT" "$REMOTE_RLRMP_ROOT"
    rsync_repo feedbax "$FEEDBAX_ROOT" "$REMOTE_FEEDBAX_ROOT"
    rsync_repo jax-cookbook "$JAX_COOKBOOK_ROOT" "$REMOTE_JAX_COOKBOOK_ROOT"
}

apply_path_patches() {
    local patch remote_file literal_from literal_to command
    while IFS= read -r patch; do
        [ -z "$patch" ] && continue
        IFS='|' read -r remote_file literal_from literal_to <<< "$patch"
        [ -n "$remote_file" ] || die "invalid path patch: $patch"
        [ -n "$literal_from" ] || die "invalid path patch: $patch"
        [ -n "$literal_to" ] || die "invalid path patch: $patch"
        command="PATCH_FROM=$(sq "$literal_from") PATCH_TO=$(sq "$literal_to") perl -0pi -e 's/\\Q\$ENV{PATCH_FROM}\\E/\$ENV{PATCH_TO}/g' $(sq "$remote_file")"
        remote_cmd "$command"
    done <<< "$PATH_PATCHES"
}

remote_nohup_sentinel() {
    local label=$1
    local workdir=$2
    local command=$3
    local done_file=$4
    local failed_file=$5
    local log_file=$6
    local remote
    remote="mkdir -p $(sq "$REMOTE_SENTINEL_DIR") $(sq "$REMOTE_RUN_DIR/logs") && rm -f $(sq "$done_file") $(sq "$failed_file") && nohup bash -lc $(sq "cd $(sq "$workdir") && { $command; touch $(sq "$done_file"); } || { touch $(sq "$failed_file"); exit 1; }") >$(sq "$log_file") 2>&1 &"
    log "starting $label"
    remote_cmd "$remote"
}

wait_for_sentinel() {
    local label=$1
    local done_file=$2
    local failed_file=$3
    if [ "$DRY_RUN" -eq 1 ]; then
        remote_cmd "test -f $(sq "$done_file") || test -f $(sq "$failed_file")"
        return 0
    fi

    local deadline
    deadline=$((SECONDS + SENTINEL_TIMEOUT_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if remote_capture "test -f $(sq "$done_file")"; then
            log "$label complete"
            return 0
        fi
        if remote_capture "test -f $(sq "$failed_file")"; then
            die "$label failed; inspect $REMOTE_RUN_DIR/logs"
        fi
        sleep "$SENTINEL_POLL_SECONDS"
    done
    die "timed out waiting for $label sentinel"
}

bootstrap_remote_env() {
    local uv_done uv_failed jax_done jax_failed
    uv_done="$REMOTE_SENTINEL_DIR/uv_sync.done"
    uv_failed="$REMOTE_SENTINEL_DIR/uv_sync.failed"
    jax_done="$REMOTE_SENTINEL_DIR/jax_cuda.done"
    jax_failed="$REMOTE_SENTINEL_DIR/jax_cuda.failed"

    remote_cmd "mkdir -p $(sq "$REMOTE_RUN_DIR/logs") $(sq "$REMOTE_SENTINEL_DIR")"
    remote_nohup_sentinel "uv sync" "$REMOTE_RLRMP_ROOT" "uv sync" \
        "$uv_done" "$uv_failed" "$REMOTE_RUN_DIR/logs/uv-sync.log"
    wait_for_sentinel "uv sync" "$uv_done" "$uv_failed"
    remote_nohup_sentinel "jax cuda install" "$REMOTE_RLRMP_ROOT" \
        "uv pip install -U \"jax[cuda12]\"" \
        "$jax_done" "$jax_failed" "$REMOTE_RUN_DIR/logs/jax-cuda-install.log"
    wait_for_sentinel "jax cuda install" "$jax_done" "$jax_failed"
}

verify_remote_device() {
    remote_cmd "cd $(sq "$REMOTE_RLRMP_ROOT") && uv run --no-sync python - <<'PY'
import jax

devices = jax.devices()
print(devices)
if not any(device.platform == 'gpu' for device in devices):
    raise SystemExit('no JAX GPU device visible')
PY"
}

sync_train_spec() {
    [ -n "$TRAIN_SPEC" ] || return 0
    local rsh
    rsh=$(rsync_rsh)
    remote_cmd "mkdir -p $(sq "$REMOTE_RUN_DIR")"
    run_cmd rsync -az --no-owner --no-group --stats -e "$rsh" \
        "$TRAIN_SPEC" "root@$SSH_HOST:$REMOTE_RUN_DIR/train-spec.json"
}

launch_training() {
    [ -n "$TRAIN_COMMAND" ] || return 0
    local train_done train_failed
    train_done="$REMOTE_SENTINEL_DIR/training.done"
    train_failed="$REMOTE_SENTINEL_DIR/training.failed"
    remote_nohup_sentinel "training" "$REMOTE_RLRMP_ROOT" "$TRAIN_COMMAND" \
        "$train_done" "$train_failed" "$REMOTE_RUN_DIR/logs/training.log"
}

main() {
    local original_args=("$@")
    first_pass_config "${original_args[@]}"
    load_config
    parse_args "${original_args[@]}"
    finalize_remote_defaults
    default_path_patches
    require_command jq
    validate_train_spec_gate
    require_real_commands

    check_docker_hub_tag
    create_pod
    wait_for_ssh_ready
    sync_repos
    apply_path_patches
    bootstrap_remote_env
    verify_remote_device
    sync_train_spec
    launch_training

    log "deploy complete for pod $POD_ID"
}

main "$@"
