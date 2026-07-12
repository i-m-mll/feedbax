#!/usr/bin/env bash
set -euo pipefail

# Deterministic RunPod deploy for the rlrmp/feedbax/jax-cookbook workflow.
#
# Training rows run as separate OS processes, so JAX's persistent compilation
# cache is enabled by default for launch commands. Rows share cache entries when
# the traced program identity matches: traced scalar values do not split cache
# classes, while static Python config, strings, booleans, object structure,
# shapes, compile options, jax/jaxlib version, and backend do. Concurrent first
# compiles of the same program all miss the cache, so row launches warm the
# first row before fanning out by default. The warm gate waits for a row log
# readiness marker or successful row completion, not the wrapper PID; set
# WARM_COMPILE_READY_REGEX / WARM_COMPILE_READY_STRING or rows-manifest
# warm_compile_ready_regex / warm_compile_ready_string fields when a training
# command emits a better signal than the default batch/step/iteration log
# pattern. Ephemeral pods do not need eviction, but the cache directory can grow
# across reused persistent volumes. Override these values with environment
# variables, `--config <file>`, or the rows manifest.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib_acquire.sh
source "$SCRIPT_DIR/lib_acquire.sh"
# shellcheck source=lib_run_prep.sh
source "$SCRIPT_DIR/lib_run_prep.sh"
# shellcheck source=lib_checkout_provenance.sh
source "$SCRIPT_DIR/lib_checkout_provenance.sh"
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
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUNPOD_VOLUME_MOUNT/jax_cache}"
RUNPOD_PORTS="${RUNPOD_PORTS:-22/tcp,8080/http}"
RUNPOD_NAME="${RUNPOD_NAME:-feedbax-rlrmp-$(date +%Y%m%d-%H%M%S)}"
RUNPOD_DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-}"
RUNPOD_ENV_JSON="${RUNPOD_ENV_JSON:-{}}"

RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-~/.runpod/ssh/RunPod-Key-Go}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
READINESS_TIMEOUT_SECONDS="${READINESS_TIMEOUT_SECONDS:-900}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-15}"
ENDPOINT_CLASSIFIER_TIMEOUT_SECONDS="${ENDPOINT_CLASSIFIER_TIMEOUT_SECONDS:-90}"
ENDPOINT_CLASSIFIER_POLL_SECONDS="${ENDPOINT_CLASSIFIER_POLL_SECONDS:-5}"
SENTINEL_TIMEOUT_SECONDS="${SENTINEL_TIMEOUT_SECONDS:-7200}"
SENTINEL_POLL_SECONDS="${SENTINEL_POLL_SECONDS:-30}"

# --- Deterministic acquisition (W1, issue c272111) ---------------------------
# Auto-rank candidate datacenters by stock for RUNPOD_GPU_ID and try one DC per
# create when RUNPOD_DATA_CENTER_IDS is unset. The endpoint-assignment grace
# window for a still-RUNNING pod must be generous (real probe: ~238s); the
# dead-state fast-fail is caught separately in seconds.
RUNPOD_DC_FALLBACK="${RUNPOD_DC_FALLBACK:-1}"
ACQUIRE_WALL_CLOCK_CAP_SECONDS="${ACQUIRE_WALL_CLOCK_CAP_SECONDS:-1800}"
ENDPOINT_GRACE_SECONDS="${ENDPOINT_GRACE_SECONDS:-420}"
DEAD_STATE_POLL_SECONDS="${DEAD_STATE_POLL_SECONDS:-5}"
MIN_BALANCE_USD="${MIN_BALANCE_USD:-1.0}"
ACQUIRE_LOCK_FILE="${ACQUIRE_LOCK_FILE:-${TMPDIR:-/tmp}/runpod_acquire_${RUNPOD_NAME}.lock}"

REMOTE_RLRMP_ROOT="${REMOTE_RLRMP_ROOT:-}"
REMOTE_FEEDBAX_ROOT="${REMOTE_FEEDBAX_ROOT:-}"
REMOTE_JAX_COOKBOOK_ROOT="${REMOTE_JAX_COOKBOOK_ROOT:-}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-}"
REMOTE_SENTINEL_DIR="${REMOTE_SENTINEL_DIR:-}"
REMOTE_ARTIFACTS_DIR="${REMOTE_ARTIFACTS_DIR:-}"
TRAIN_COMMAND="${TRAIN_COMMAND:-}"
RUNPOD_RUN_CONFIG_FILE="${RUNPOD_RUN_CONFIG_FILE:-}"
REMOTE_DEPLOY_CONFIG_PATH="${REMOTE_DEPLOY_CONFIG_PATH:-}"

# --- Parallel multi-row launch (W2, issue 9ca50b1) ---------------------------
ROWS_MANIFEST="${ROWS_MANIFEST:-}"
ROW_LAUNCH_STAGGER_SECONDS="${ROW_LAUNCH_STAGGER_SECONDS:-10}"
MAX_PARALLEL_ROWS="${MAX_PARALLEL_ROWS:-4}"
WARM_COMPILE_FIRST="${WARM_COMPILE_FIRST:-1}"
WARM_COMPILE_READY_REGEX="${WARM_COMPILE_READY_REGEX:-([Bb]atch|[Ss]tep|[Ii]ter|[Ii]t)[[:space:]=:]+[0-9]+}"
WARM_COMPILE_READY_STRING="${WARM_COMPILE_READY_STRING:-}"

# Acquisition bookkeeping.
ACQUIRED_DC=""
POD_CREATED_BY_US=0
ACQUIRE_LOCK_HELD=0

# Per-run unique name prefix for create-leak sweeping (FIX 1). Every `pod create`
# attempt gets a UNIQUE name that starts with this prefix, so a name-based
# `pod list` sweep can find and tear down any pod created server-side even when
# the CLI result was nonzero/unparseable or a signal interrupted the create.
RUN_TAG="${RUN_TAG:-${RUNPOD_NAME}-r$$-$(date +%s)}"
# Monotonic counter feeding the per-attempt suffix (no Math.random dependency).
POD_NAME_COUNTER=0
# Name we passed to the most recent create attempt (may have leaked a pod even
# if POD_ID was never parsed).
LAST_CREATE_NAME=""
# Idempotency guard so EXIT cleanup does not double-run after a signal handler
# already ran it.
ACQUIRE_CLEANUP_DONE=0

CONFIG_FILE=""
DRY_RUN=0
ACQUIRE_ONLY=0
POD_ID="${POD_ID:-}"
TRAIN_SPEC="${TRAIN_SPEC:-}"
SKIP_IMAGE_CHECK=0

SSH_HOST=""
SSH_PORT=""
PROVIDED_ENDPOINT=0
ENDPOINT_SOURCE="missing"
ENDPOINT_CLASSIFICATION="missing_direct_endpoint"
SSH_ERROR=""

usage() {
    cat <<'USAGE'
Usage: scripts/deploy/runpod_deploy.sh [options]

Options:
  --config <file>           Source a bash config file after defaults.
  --dry-run                 Print commands without executing them.
  --acquire-only            Create/attach/probe SSH+GPU and exit before deploy.
  --pod-id <id>             Reuse an existing RunPod pod instead of creating one.
  --ssh-host <host>         Use an already-known direct SSH host; skips discovery.
  --ssh-port <port>         Use an already-known direct SSH port; skips discovery.
  --train-spec <json>       Required when a training launch command is configured.
  --launch-command <cmd>    Remote training command to launch after bootstrap.
  --rows-manifest <json>    Launch N training rows from a rows-manifest instead
                            of a single --launch-command.
  --skip-image-check        Skip Docker Hub tag existence check.
  -h, --help                Show this help.

Acquisition: when --pod-id / --ssh-host are not given and RUNPOD_DATA_CENTER_IDS
is empty, candidate datacenters are ranked by stock for RUNPOD_GPU_ID and tried
one-per-create. Dead-state pods (EXITED/TERMINATED) are torn down within seconds;
a still-RUNNING pod gets ENDPOINT_GRACE_SECONDS to assign an endpoint. A trap
tears down any pod this script created on every non-acquired exit path.

The training launch is refused unless --train-spec points to JSON containing
`"user_confirmed": true`; failure prints the spec table and exits 2.

On reused pods (--pod-id or provided SSH endpoint), deploy probes the existing
remote `.venv` with `uv run --no-sync python -c 'import jax, jax.numpy; ...'`
before any install step. Probe success skips environment installation; probe
failure clears and rebuilds the venv, then reruns the normal sync/CUDA-JAX
install sequence.
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
            --acquire-only)
                ACQUIRE_ONLY=1
                ;;
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
            --rows-manifest)
                shift
                [ "$#" -gt 0 ] || die "--rows-manifest requires a JSON file"
                ROWS_MANIFEST=$1
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
    if ! has_provided_endpoint; then
        require_command curl
        require_command runpodctl
    fi
    require_command jq
    require_command ssh
    if [ "$ACQUIRE_ONLY" -eq 0 ]; then
        require_command perl
        require_command rsync
    fi
}

has_provided_endpoint() {
    [ "$PROVIDED_ENDPOINT" -eq 1 ]
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
    REMOTE_ARTIFACTS_DIR="${REMOTE_ARTIFACTS_DIR:-$REMOTE_RLRMP_ROOT/_artifacts}"
    RUNPOD_RUN_CONFIG_FILE="${RUNPOD_RUN_CONFIG_FILE:-$FEEDBAX_ROOT/.runpod/run-config.json}"
    REMOTE_DEPLOY_CONFIG_PATH="${REMOTE_DEPLOY_CONFIG_PATH:-$RUNPOD_VOLUME_MOUNT/feedbax_runs/runpod-deploy-config.json}"
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

# acquire_status emits the machine-readable acquisition vocabulary, one line
# per poll/event: provisioning / endpoint_assigned / ssh_ready / dead_exited /
# attempt_timeout / acquired. Parseable without jq.
acquire_status() {
    local state=$1
    local dc=${2:-${ACQUIRED_DC:-none}}
    printf 'acquire_status=%s pod=%s dc=%s ssh_host=%s ssh_port=%s\n' \
        "$state" "${POD_ID:-none}" "${dc:-none}" "${SSH_HOST:-none}" "${SSH_PORT:-none}" >&2
}

acquire_lock_acquire() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    local dir
    dir=$(dirname "$ACQUIRE_LOCK_FILE")
    mkdir -p "$dir" 2>/dev/null || true
    # mkdir is atomic: a directory lock prevents two loops racing the same run.
    if mkdir "$ACQUIRE_LOCK_FILE" 2>/dev/null; then
        ACQUIRE_LOCK_HELD=1
        printf '%s\n' "$$" >"$ACQUIRE_LOCK_FILE/pid" 2>/dev/null || true
        return 0
    fi
    die "acquisition lock held: $ACQUIRE_LOCK_FILE (another deploy loop is running for $RUNPOD_NAME)"
}

acquire_lock_release() {
    [ "$ACQUIRE_LOCK_HELD" -eq 1 ] || return 0
    rm -rf "$ACQUIRE_LOCK_FILE" 2>/dev/null || true
    ACQUIRE_LOCK_HELD=0
}

# teardown_pod removes a pod by id. Robust against word-splitting: the id is a
# single quoted scalar and we never iterate an unquoted array (a recent manual
# probe lost teardown to shell word-splitting, so keep this defensive).
teardown_pod() {
    local pod_id=${1:-}
    [ -n "$pod_id" ] || return 0
    [ "$DRY_RUN" -eq 1 ] && { log "dry-run: would tear down pod $pod_id"; return 0; }
    log "tearing down pod $pod_id"
    runpodctl pod remove "$pod_id" >/dev/null 2>&1 ||
        runpodctl pod stop "$pod_id" >/dev/null 2>&1 ||
        log "warning: teardown of pod $pod_id reported a non-zero status"
}

# next_pod_name returns a UNIQUE pod name for the next create attempt (FIX 1).
# The randomness is derived from $$, the epoch, and a monotonic counter — no
# Math.random / external entropy. The name starts with RUN_TAG so the sweep can
# find every pod this run created by prefix.
next_pod_name() {
    local dc=${1:-auto}
    dc=$(printf '%s' "$dc" | tr -cd '[:alnum:]_.-')
    [ -n "$dc" ] || dc=auto
    POD_NAME_COUNTER=$((POD_NAME_COUNTER + 1))
    local rand
    rand="$$$(date +%s)$POD_NAME_COUNTER"
    printf '%s-%s-%s-%s' "$RUN_TAG" "$dc" "$POD_NAME_COUNTER" "$rand"
}

# sweep_created_pods lists pods by this run's unique name prefix and tears down
# any match we are NOT keeping (FIX 1). It runs after every create-attempt
# outcome (success, nonzero, unparseable) and in cleanup, so a pod created
# server-side with no parseable id — or one created right before a signal — is
# still found and removed. Safe and idempotent when the list is empty.
#   $1 (optional) pod id to KEEP (e.g. the acquired pod); empty keeps nothing.
sweep_created_pods() {
    local keep=${1:-}
    [ "$DRY_RUN" -eq 1 ] && return 0
    command -v runpodctl >/dev/null 2>&1 || return 0
    [ -n "$RUN_TAG" ] || return 0
    local list_json pid
    list_json=$(runpodctl pod list --output json 2>/dev/null) || return 0
    [ -n "$list_json" ] || return 0
    while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        [ -n "$keep" ] && [ "$pid" = "$keep" ] && continue
        log "sweep: tearing down leaked pod $pid (name prefix $RUN_TAG)"
        teardown_pod "$pid"
    done < <(printf '%s' "$list_json" | pod_ids_by_name_prefix "$RUN_TAG")
}

# acquire_cleanup tears down any pod this run created/leaked and releases the
# lock. Idempotent: guarded so the EXIT handler does not repeat work a signal
# handler already did.  $1 is the pod id to KEEP (the acquired pod), if any.
ACQUIRED=0
acquire_cleanup() {
    [ "$ACQUIRE_CLEANUP_DONE" -eq 1 ] && return 0
    ACQUIRE_CLEANUP_DONE=1
    local keep=${1:-}
    # Tear down a parsed-but-unacquired pod first (fast path), then sweep by name
    # to catch any server-side pod whose id we never parsed.
    if [ "$ACQUIRED" -eq 0 ] && [ -n "${POD_ID:-}" ]; then
        teardown_pod "$POD_ID"
    fi
    if [ "$ACQUIRED" -eq 1 ]; then
        keep=${keep:-${POD_ID:-}}
    fi
    sweep_created_pods "$keep"
    acquire_lock_release
}

# acquire_trap fires on EXIT. If we created a pod but never reached the acquired
# state, tear it down so a failed/timed-out attempt never leaves a billable pod
# running. Always releases the lock. Idempotent via acquire_cleanup's guard.
acquire_trap() {
    local rc=$?
    acquire_cleanup
    return "$rc"
}

# acquire_signal_trap handles INT/TERM (FIX 2): run cleanup, then EXIT with the
# conventional signal status (130 INT, 143 TERM) so no acquisition/deploy code
# can resume after a handled signal. EXIT cleanup is a no-op afterward (guarded).
acquire_signal_trap() {
    local signame=$1
    log "received SIG$signame during acquisition; cleaning up and exiting"
    acquire_cleanup
    case "$signame" in
        INT) exit 130 ;;
        TERM) exit 143 ;;
        *) exit 1 ;;
    esac
}

# precheck_balance fails CLOSED (FIX 5): if the balance cannot be read or parsed
# (transport/auth/CLI failure, or a non-numeric balance), refuse to create the
# pod rather than bypassing the cost guard. Set ALLOW_UNKNOWN_BALANCE=1 to opt
# out deliberately (then it only warns); the default is fail-closed.
ALLOW_UNKNOWN_BALANCE="${ALLOW_UNKNOWN_BALANCE:-0}"
precheck_balance() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    has_provided_endpoint && return 0
    [ -n "$POD_ID" ] && return 0
    local user_json result
    if ! user_json=$(capture_cmd runpodctl user --output json 2>/dev/null); then
        if [ "$ALLOW_UNKNOWN_BALANCE" -eq 1 ]; then
            log "warning: could not read account balance; ALLOW_UNKNOWN_BALANCE=1, continuing"
            return 0
        fi
        die "account balance could not be read (runpodctl user failed); refusing to create a pod (set ALLOW_UNKNOWN_BALANCE=1 to override)"
    fi
    # check_balance_floor: rc 0 ok, rc 1 below floor, rc 2 unparseable balance.
    if result=$(printf '%s' "$user_json" | check_balance_floor "$MIN_BALANCE_USD"); then
        log "balance precheck $result"
        return 0
    fi
    if [ "$result" = "unknown" ] && [ "$ALLOW_UNKNOWN_BALANCE" -eq 1 ]; then
        log "warning: account balance unparseable; ALLOW_UNKNOWN_BALANCE=1, continuing"
        return 0
    fi
    die "account balance precheck failed: $result (floor \$$MIN_BALANCE_USD); refusing to create a pod"
}

# rank_candidate_dcs prints the ordered DC list for the create loop, one per
# line. Honors an explicit RUNPOD_DATA_CENTER_IDS override; otherwise ranks by
# stock for RUNPOD_GPU_ID when RUNPOD_DC_FALLBACK is enabled. Falls back to a
# single empty entry (let RunPod pick) when ranking yields nothing.
rank_candidate_dcs() {
    if [ -n "$RUNPOD_DATA_CENTER_IDS" ]; then
        printf '%s\n' "$RUNPOD_DATA_CENTER_IDS" | tr ',' '\n'
        return 0
    fi
    if [ "$RUNPOD_DC_FALLBACK" -ne 1 ] || [ "$DRY_RUN" -eq 1 ]; then
        printf '\n'
        return 0
    fi
    local dc_json ranked
    dc_json=$(capture_cmd runpodctl datacenter list --output json 2>/dev/null) || {
        log "warning: datacenter list failed; letting RunPod choose"
        printf '\n'
        return 0
    }
    ranked=$(printf '%s' "$dc_json" | rank_datacenters_for_gpu "$RUNPOD_GPU_ID")
    if [ -z "$ranked" ]; then
        log "warning: no datacenter reports stock for $RUNPOD_GPU_ID; letting RunPod choose"
        printf '\n'
        return 0
    fi
    printf '%s\n' "$ranked"
}

# create_pod_in_dc issues one `pod create`, optionally pinned to a single DC.
# Sets POD_ID and POD_CREATED_BY_US on success. Returns non-zero if the create
# call itself fails (so the loop can advance to the next DC).
create_pod_in_dc() {
    local dc=${1:-}
    local cmd output pod_id pod_name rc
    # Unique per-attempt name so the name-based sweep can find this pod even if
    # the create result is nonzero/unparseable or a signal interrupts us (FIX 1).
    pod_name=$(next_pod_name "$dc")
    LAST_CREATE_NAME=$pod_name
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
        --name "$pod_name"
        --env "$RUNPOD_ENV_JSON"
        --output json
    )
    if [ -n "$dc" ]; then
        cmd+=(--data-center-ids "$dc")
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        run_cmd "${cmd[@]}"
        POD_ID="dry-run-pod"
        POD_CREATED_BY_US=1
        ACQUIRED_DC=${dc:-dry-run-dc}
        return 0
    fi

    # Capture the create result without letting a nonzero exit abort the script;
    # a server-side pod may exist even on a nonzero/unparseable CLI result, so we
    # must sweep by name afterward regardless of the outcome.
    set +e
    output=$(capture_cmd "${cmd[@]}")
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        # The create may still have created a pod server-side; sweep by name.
        sweep_created_pods
        return 1
    fi
    pod_id=$(printf '%s\n' "$output" |
        jq -r '(.id // .pod.id // .podId // .data.id // .data.pod.id // empty)')
    if [ -z "$pod_id" ]; then
        # Unparseable result but the pod may exist server-side under our unique
        # name; sweep so it is not leaked.
        sweep_created_pods
        return 1
    fi
    POD_ID=$pod_id
    POD_CREATED_BY_US=1
    ACQUIRED_DC=${dc:-auto}
    log "created pod $POD_ID${dc:+ in $dc} (name $pod_name)"
    return 0
}

# wait_for_endpoint_or_dead polls a freshly-created pod until it either assigns
# an endpoint (rc 0, SSH_HOST/SSH_PORT set), flips to a dead state (rc 3), or
# exceeds the per-attempt grace window (rc 1). Dead-state is caught in seconds;
# the grace window only governs the still-RUNNING/no-endpoint case.
wait_for_endpoint_or_dead() {
    [ "$DRY_RUN" -eq 1 ] && {
        run_cmd runpodctl pod get "$POD_ID" --output json
        SSH_HOST="dry-run-host"; SSH_PORT="22"
        ENDPOINT_SOURCE="ssh_object"
        ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
        acquire_status endpoint_assigned
        return 0
    }
    local deadline detail class rest ip port
    deadline=$((SECONDS + ENDPOINT_GRACE_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        detail=$(capture_cmd runpodctl pod get "$POD_ID" --output json) || {
            sleep "$DEAD_STATE_POLL_SECONDS"; continue
        }
        class=$(printf '%s' "$detail" | classify_pod_state)
        case "$class" in
            ready\ *)
                read -r _ ip port <<<"$class"
                SSH_HOST=$ip; SSH_PORT=$port
                ENDPOINT_SOURCE="ssh_object"
                ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
                SSH_ERROR="none"
                acquire_status endpoint_assigned
                return 0
                ;;
            dead\ *)
                rest=${class#dead }
                SSH_ERROR=$(safe_status_value "$rest")
                acquire_status dead_exited
                return 3
                ;;
            *)
                acquire_status provisioning
                sleep "$DEAD_STATE_POLL_SECONDS"
                ;;
        esac
    done
    acquire_status attempt_timeout
    return 1
}

# write_boot_heartbeat drops an in-pod heartbeat after the first successful SSH
# so the compile phase stays distinguishable from death: a live but silent pod
# keeps a recent heartbeat mtime, a dead one does not.
write_boot_heartbeat() {
    [ "$DRY_RUN" -eq 1 ] && return 0
    remote_cmd "mkdir -p $(sq "$REMOTE_RUN_DIR") && date -u +%Y-%m-%dT%H:%M:%SZ > $(sq "$REMOTE_RUN_DIR/boot_heartbeat")" ||
        log "warning: could not write boot heartbeat"
}

# acquire_pod is the deterministic, watched acquisition phase. It iterates the
# ranked datacenters one-create-at-a-time, fast-fails dead pods within seconds,
# grants a generous endpoint grace to still-RUNNING pods, proves SSH+GPU, and is
# bounded by a hard wall-clock cap. The acquire_trap guarantees teardown of any
# pod we created on every non-acquired exit path.
acquire_pod() {
    if [ -n "$POD_ID" ]; then
        log "using existing pod $POD_ID"
        if ! wait_for_ssh_ready; then
            die "provided pod $POD_ID did not become ssh-ready"
        fi
        ACQUIRED=1
        acquire_status acquired
        return 0
    fi

    acquire_lock_acquire
    precheck_balance

    local wall_deadline candidates dc attempted=0
    wall_deadline=$((SECONDS + ACQUIRE_WALL_CLOCK_CAP_SECONDS))
    read_lines_into candidates < <(rank_candidate_dcs)
    log "candidate datacenters: ${candidates[*]:-<runpod-choice>}"

    for dc in "${candidates[@]}"; do
        if [ "$SECONDS" -ge "$wall_deadline" ]; then
            die "acquisition wall-clock cap (${ACQUIRE_WALL_CLOCK_CAP_SECONDS}s) reached before acquiring a pod"
        fi
        attempted=$((attempted + 1))
        acquire_status provisioning "${dc:-runpod-choice}"
        if ! create_pod_in_dc "$dc"; then
            log "create failed for dc=${dc:-<runpod-choice>}; trying next"
            POD_ID=""; POD_CREATED_BY_US=0
            continue
        fi

        if ! wait_for_endpoint_or_dead; then
            # attempt_timeout or dead: tear down THIS pod, advance to next DC.
            teardown_pod "$POD_ID"
            POD_ID=""; POD_CREATED_BY_US=0; SSH_HOST=""; SSH_PORT=""
            continue
        fi

        if probe_ssh_gpu; then
            ENDPOINT_CLASSIFICATION="direct_endpoint_ready"
            SSH_ERROR="none"
            acquire_status ssh_ready
            write_boot_heartbeat
            ACQUIRED=1
            acquire_status acquired
            # Keep this pod; sweep any earlier-attempt pods that leaked.
            sweep_created_pods "$POD_ID"
            log "acquired pod $POD_ID in ${ACQUIRED_DC} after $attempted attempt(s)"
            return 0
        fi

        log "pod $POD_ID assigned an endpoint but nvidia-smi probe failed; tearing down"
        teardown_pod "$POD_ID"
        POD_ID=""; POD_CREATED_BY_US=0; SSH_HOST=""; SSH_PORT=""
    done

    die "could not acquire a working pod after $attempted attempt(s) across candidate datacenters"
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

extract_ssh_error() {
    jq -r '
        .ssh as $ssh
        | (
            if ($ssh | type) == "object" then
                ($ssh.error // $ssh.message // $ssh.reason // empty)
            else
                empty
            end
          ) // .sshError // .runtime.sshError // empty
    '
}

safe_status_value() {
    local value=${1:-none}
    value=${value:-none}
    printf '%s\n' "$value" | tr '[:space:]' '_' | tr -cd '[:alnum:]_.:/=@,+-'
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

clear_discovered_endpoint() {
    if [ "$ENDPOINT_SOURCE" != "provided" ]; then
        SSH_HOST=""
        SSH_PORT=""
    fi
}

classify_endpoint_detail() {
    local detail=$1
    local ssh_ip ssh_port ssh_command ssh_error
    if has_provided_endpoint; then
        ENDPOINT_SOURCE="provided"
        ENDPOINT_CLASSIFICATION="known_direct_endpoint"
        SSH_ERROR="none"
        return 0
    fi

    clear_discovered_endpoint
    ssh_ip=$(printf '%s\n' "$detail" | extract_ssh_field ip)
    ssh_port=$(printf '%s\n' "$detail" | extract_ssh_field port)
    if [ -z "$ssh_ip" ]; then
        ssh_ip=$(printf '%s\n' "$detail" | extract_ssh_field host)
    fi
    if [ -n "$ssh_ip" ]; then
        SSH_HOST=$ssh_ip
    fi
    if [ -n "$ssh_port" ]; then
        SSH_PORT=$ssh_port
    fi

    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        ENDPOINT_SOURCE="ssh_object"
        ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
        SSH_ERROR="none"
        return 0
    fi

    ssh_command=$(printf '%s\n' "$detail" | extract_ssh_command)
    parse_ssh_command_fields "$ssh_command"
    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        ENDPOINT_SOURCE="ssh_command"
        ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
        SSH_ERROR="none"
        return 0
    fi

    ssh_error=$(printf '%s\n' "$detail" | extract_ssh_error)
    SSH_ERROR=$(safe_status_value "${ssh_error:-missing_ssh_metadata}")
    if [ -n "$SSH_HOST" ] || [ -n "$SSH_PORT" ] || [ -n "$ssh_command" ]; then
        ENDPOINT_SOURCE="partial"
        ENDPOINT_CLASSIFICATION="partial_direct_endpoint"
    else
        ENDPOINT_SOURCE="missing"
        ENDPOINT_CLASSIFICATION="missing_direct_endpoint"
    fi
}

print_endpoint_info() {
    printf 'pod_id=%s\n' "${POD_ID:-none}"
    printf 'endpoint_source=%s\n' "$ENDPOINT_SOURCE"
    printf 'endpoint_classification=%s\n' "$ENDPOINT_CLASSIFICATION"
    printf 'ssh_host=%s\n' "${SSH_HOST:-none}"
    printf 'ssh_port=%s\n' "${SSH_PORT:-none}"
    printf 'ssh_key=%s\n' "$(expand_path "$RUNPOD_SSH_KEY")"
    printf 'ssh_error=%s\n' "${SSH_ERROR:-none}"
}

remote_cmd() {
    local command=$1
    local key_path
    key_path=$(expand_path "$RUNPOD_SSH_KEY")
    local cmd=(ssh -n
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
    local cmd=(ssh -n
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
    if has_provided_endpoint; then
        ENDPOINT_SOURCE="provided"
        ENDPOINT_CLASSIFICATION="known_direct_endpoint"
        SSH_ERROR="none"
        probe_ssh_gpu
        ENDPOINT_CLASSIFICATION="direct_endpoint_ready"
        log "known ssh endpoint and nvidia-smi are ready"
        return 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        run_cmd runpodctl pod get "$POD_ID" --output json
        SSH_HOST="dry-run-host"
        SSH_PORT="22"
        ENDPOINT_SOURCE="ssh_object"
        ENDPOINT_CLASSIFICATION="direct_endpoint_discovered"
        SSH_ERROR="none"
        probe_ssh_gpu
        ENDPOINT_CLASSIFICATION="direct_endpoint_ready"
        return 0
    fi

    local deadline classifier_deadline detail poll_seconds pod_class
    deadline=$((SECONDS + READINESS_TIMEOUT_SECONDS))
    classifier_deadline=$((SECONDS + ENDPOINT_CLASSIFIER_TIMEOUT_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        detail=$(capture_cmd runpodctl pod get "$POD_ID" --output json)
        pod_class=$(printf '%s' "$detail" | classify_pod_state)
        if [[ $pod_class == dead\ * ]]; then
            SSH_ERROR=$(safe_status_value "${pod_class#dead }")
            print_endpoint_info >&2
            die "pod $POD_ID entered dead state (${pod_class#dead })"
        fi
        classify_endpoint_detail "$detail"
        log "endpoint source=$ENDPOINT_SOURCE classification=$ENDPOINT_CLASSIFICATION ssh_error=${SSH_ERROR:-none}"
        if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
            if [ "$SECONDS" -ge "$classifier_deadline" ]; then
                print_endpoint_info >&2
                die "no direct ssh endpoint after ${ENDPOINT_CLASSIFIER_TIMEOUT_SECONDS}s; not waiting full readiness timeout"
            fi
            poll_seconds=$ENDPOINT_CLASSIFIER_POLL_SECONDS
            if [ "$poll_seconds" -le 0 ]; then
                poll_seconds=1
            fi
            sleep "$poll_seconds"
            continue
        fi
        if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && probe_ssh_gpu; then
            ENDPOINT_CLASSIFICATION="direct_endpoint_ready"
            SSH_ERROR="none"
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
    # rsync carries its protocol over SSH stdin, so unlike command-only SSH
    # calls this transport must not use ``-n``.
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
        --exclude /_artifacts \
        --exclude __pycache__ \
        --exclude .pytest_cache \
        --exclude .mypy_cache \
        --exclude .ruff_cache \
        --exclude .DS_Store \
        --exclude worktrees \
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

# repair_remote_artifacts fixes the _artifacts path on the pod after repo sync.
# Code rsync excludes top-level _artifacts so remote checkpoint/run state is not
# deleted or replaced. This repair remains for fresh pods and for older pods
# that may already contain a dangling shared-worktree symlink.
repair_remote_artifacts() {
    local path=$REMOTE_ARTIFACTS_DIR
    [ -n "$path" ] || return 0
    local snippet
    # shellcheck disable=SC2016
    snippet='p='"$(sq "$path")"'
if [ -L "$p" ] && [ ! -e "$p" ]; then
    echo "artifacts: dangling symlink -> replacing with real dir" >&2
    rm -f "$p" && mkdir -p "$p"
elif [ -L "$p" ] && [ -e "$p" ]; then
    echo "artifacts: valid symlink, leaving in place" >&2
elif [ -d "$p" ]; then
    echo "artifacts: real dir, leaving in place" >&2
elif [ -e "$p" ]; then
    echo "artifacts: WARNING non-dir file present at $p; leaving in place" >&2
else
    mkdir -p "$p"
fi'
    remote_cmd "$snippet"
}

bootstrap_remote_env() {
    local uv_done uv_failed jax_done jax_failed
    uv_done="$REMOTE_SENTINEL_DIR/uv_sync.done"
    uv_failed="$REMOTE_SENTINEL_DIR/uv_sync.failed"
    jax_done="$REMOTE_SENTINEL_DIR/jax_cuda.done"
    jax_failed="$REMOTE_SENTINEL_DIR/jax_cuda.failed"

    remote_cmd "mkdir -p $(sq "$REMOTE_RUN_DIR/logs") $(sq "$REMOTE_SENTINEL_DIR")"
    remote_nohup_sentinel "uv sync" "$REMOTE_RLRMP_ROOT" "${1:-uv sync}" \
        "$uv_done" "$uv_failed" "$REMOTE_RUN_DIR/logs/uv-sync.log"
    wait_for_sentinel "uv sync" "$uv_done" "$uv_failed"
    remote_nohup_sentinel "jax cuda install" "$REMOTE_RLRMP_ROOT" \
        "uv pip install -U \"jax[cuda12]\"" \
        "$jax_done" "$jax_failed" "$REMOTE_RUN_DIR/logs/jax-cuda-install.log"
    wait_for_sentinel "jax cuda install" "$jax_done" "$jax_failed"
}

mark_bootstrap_branch() {
    local branch=$1
    local marker_file="$REMOTE_SENTINEL_DIR/${branch}.done"
    local line="venv_probe_branch=$branch"
    log "$line"
    remote_cmd "mkdir -p $(sq "$REMOTE_SENTINEL_DIR") $(sq "$REMOTE_RUN_DIR/logs") && printf '%s\n' $(sq "$line") >> $(sq "$REMOTE_RUN_DIR/logs/venv-probe.log") && touch $(sq "$marker_file")"
}

clear_venv_probe_markers() {
    remote_cmd "rm -f $(sq "$REMOTE_SENTINEL_DIR/venv_probe.done") $(sq "$REMOTE_SENTINEL_DIR/venv_probe.failed") $(sq "$REMOTE_SENTINEL_DIR/probe_ok.done") $(sq "$REMOTE_SENTINEL_DIR/probe_failed_rebuilding.done") $(sq "$REMOTE_SENTINEL_DIR/rebuild_done.done")"
}

probe_reused_remote_env() {
    local probe_done probe_failed
    probe_done="$REMOTE_SENTINEL_DIR/venv_probe.done"
    probe_failed="$REMOTE_SENTINEL_DIR/venv_probe.failed"
    remote_nohup_sentinel "venv consistency probe" "$REMOTE_RLRMP_ROOT" \
        "uv run --no-sync python -c 'import jax, jax.numpy; print(jax.devices())'" \
        "$probe_done" "$probe_failed" "$REMOTE_RUN_DIR/logs/venv-probe.log"
    wait_for_sentinel_result "venv consistency probe" "$probe_done" "$probe_failed"
}

bootstrap_remote_env_for_pod() {
    remote_cmd "mkdir -p $(sq "$REMOTE_RUN_DIR/logs") $(sq "$REMOTE_SENTINEL_DIR")"
    if [ "$POD_CREATED_BY_US" -eq 1 ]; then
        bootstrap_remote_env
        return 0
    fi

    clear_venv_probe_markers
    if probe_reused_remote_env; then
        mark_bootstrap_branch "probe_ok"
        return 0
    fi

    mark_bootstrap_branch "probe_failed_rebuilding"
    bootstrap_remote_env "uv venv --clear && uv sync"
    mark_bootstrap_branch "rebuild_done"
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

main() {
    first_pass_config "$@"
    load_config
    emit_checkout_provenance "$SCRIPT_DIR/${BASH_SOURCE[0]##*/}" "$FEEDBAX_ROOT" "$RLRMP_ROOT"
    parse_args "$@"
    if { [ -n "$SSH_HOST" ] && [ -z "$SSH_PORT" ]; } ||
        { [ -z "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; }; then
        die "--ssh-host and --ssh-port must be provided together"
    fi
    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        PROVIDED_ENDPOINT=1
    fi
    finalize_remote_defaults
    default_path_patches
    require_command jq
    if [ "$ACQUIRE_ONLY" -eq 0 ]; then
        validate_train_spec_gate
        validate_declared_baselines
        validate_training_manifest_preflight
    fi
    write_run_config
    require_real_commands

    # Trap guarantees teardown of any pod THIS script created/leaked on every
    # non-acquired exit path, and always releases the acquisition lock. INT/TERM
    # get a dedicated handler that cleans up then EXITS with the conventional
    # signal status (FIX 2), so no acquisition/deploy code can resume after a
    # handled signal; the EXIT handler is idempotent (acquire_cleanup guard).
    trap 'acquire_trap' EXIT
    trap 'acquire_signal_trap INT' INT
    trap 'acquire_signal_trap TERM' TERM

    if has_provided_endpoint; then
        log "using provided ssh endpoint $SSH_HOST:$SSH_PORT"
        wait_for_ssh_ready
        ACQUIRED=1
    else
        check_docker_hub_tag
        acquire_pod
    fi

    if [ "$ACQUIRE_ONLY" -eq 1 ]; then
        print_endpoint_info
        log "acquire-only complete for pod ${POD_ID:-<provided-endpoint>}"
        exit 0
    fi
    sync_repos
    repair_remote_artifacts
    sync_declared_baselines
    validate_remote_declared_baselines
    apply_path_patches
    bootstrap_remote_env_for_pod
    verify_remote_device
    verify_remote_resumes
    write_run_config
    sync_run_config
    sync_train_spec
    launch_training

    log "deploy complete for pod $POD_ID"
}

main "$@"
