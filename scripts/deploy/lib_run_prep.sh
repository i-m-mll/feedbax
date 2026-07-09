#!/usr/bin/env bash
# Sourceable run-preparation helpers shared by deploy providers. The default
# provider is RunPod and delegates transport to the driver-defined SSH/rsync
# helpers so existing runpod_deploy.sh CLI behavior stays unchanged.

run_prep_provider() {
    printf '%s\n' "${RUN_PREP_PROVIDER:-runpod}"
}

provider_workdir() {
    case "$(run_prep_provider)" in
        runpod)
            printf '%s\n' "$REMOTE_RLRMP_ROOT"
            ;;
        local)
            printf '%s\n' "${LOCAL_RUN_PREP_WORKDIR:-${REMOTE_RLRMP_ROOT:-$PWD}}"
            ;;
        *)
            die "unknown run-prep provider: $(run_prep_provider)"
            ;;
    esac
}

provider_exec() {
    local command=$1
    case "$(run_prep_provider)" in
        runpod)
            remote_cmd "$command"
            ;;
        local)
            run_cmd bash -lc "$command"
            ;;
        *)
            die "unknown run-prep provider: $(run_prep_provider)"
            ;;
    esac
}

provider_capture() {
    local command=$1
    case "$(run_prep_provider)" in
        runpod)
            remote_capture "$command"
            ;;
        local)
            capture_cmd bash -lc "$command"
            ;;
        *)
            die "unknown run-prep provider: $(run_prep_provider)"
            ;;
    esac
}

provider_copy() {
    local source=$1
    local target=$2
    local delete=${3:-0}
    local rsh
    case "$(run_prep_provider)" in
        runpod)
            rsh=$(rsync_rsh)
            if [ "$delete" -eq 1 ]; then
                run_cmd rsync -az --delete --no-owner --no-group --stats -e "$rsh" \
                    "$source" "root@$SSH_HOST:$target"
            else
                run_cmd rsync -az --no-owner --no-group --stats -e "$rsh" \
                    "$source" "root@$SSH_HOST:$target"
            fi
            ;;
        local)
            if [ "$delete" -eq 1 ]; then
                rm -rf "$target"
            fi
            mkdir -p "$(dirname "$target")"
            if [ -d "${source%/}" ]; then
                mkdir -p "$target"
                if [ "${source%/}" != "$source" ]; then
                    run_cmd bash -lc "shopt -s dotglob nullglob; cp -a $(sq "${source%/}")/* $(sq "$target")/"
                else
                    run_cmd cp -a "$source" "$target"
                fi
            else
                run_cmd cp -a "$source" "$target"
            fi
            ;;
        *)
            die "unknown run-prep provider: $(run_prep_provider)"
            ;;
    esac
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
    [ -n "$TRAIN_COMMAND" ] || [ -n "$ROWS_MANIFEST" ] || return 0
    if [ -z "$TRAIN_SPEC" ] || [ ! -f "$TRAIN_SPEC" ]; then
        print_spec_table "$TRAIN_SPEC"
        exit 2
    fi
    if ! jq -e '.user_confirmed == true' "$TRAIN_SPEC" >/dev/null; then
        print_spec_table "$TRAIN_SPEC"
        exit 2
    fi
}

baseline_jq_filter() {
    cat <<'JQ'
def baseline_entries($label):
  [
    (.baseline_checkpoint_path // .baseline_checkpoint // .checkpoint_path // empty) as $path
    | (.baseline_completed_batches // .baseline_completed_batch // .completed_batches // .completed_batch // empty) as $batch
    | select(($path | tostring) != "")
    | {path: ($path | tostring), completed_batch: ($batch | tostring), label: $label}
  ]
  +
  [
    (.resume // {}) as $resume
    | ($resume.baseline_checkpoint_path // $resume.baseline_checkpoint // $resume.checkpoint_path // $resume.checkpoint // empty) as $path
    | ($resume.baseline_completed_batches // $resume.baseline_completed_batch // $resume.completed_batches // $resume.completed_batch // empty) as $batch
    | select(($path | tostring) != "")
    | {path: ($path | tostring), completed_batch: ($batch | tostring), label: $label}
  ];

if type != "object" then []
elif has("rows") then
  [ .rows[]
    | baseline_entries("row:" + ((.id // "unknown") | tostring))[]
  ]
else
  baseline_entries($source_label)
end
JQ
}

declared_baselines_json() {
    local entries="[]"
    if [ -n "$TRAIN_SPEC" ] && [ -f "$TRAIN_SPEC" ]; then
        entries=$(jq --arg source_label train_spec "$(baseline_jq_filter)" "$TRAIN_SPEC")
    fi
    if [ -n "$ROWS_MANIFEST" ] && [ -f "$ROWS_MANIFEST" ]; then
        entries=$(jq -s 'add' \
            <(printf '%s\n' "$entries") \
            <(jq --arg source_label rows_manifest "$(baseline_jq_filter)" "$ROWS_MANIFEST"))
    fi
    printf '%s\n' "$entries"
}

baseline_source_path() {
    local source=$1
    if [[ $source = /* ]]; then
        printf '%s\n' "$source"
    else
        printf '%s/%s\n' "$RLRMP_ROOT" "$source"
    fi
}

baseline_remote_target() {
    local source=$1 source_path rel base
    source_path=$(baseline_source_path "$source")
    case "$source" in
        "$RLRMP_ROOT/_artifacts"/*)
            rel=${source#"$RLRMP_ROOT/_artifacts"/}
            printf '%s/%s\n' "$REMOTE_ARTIFACTS_DIR" "$rel"
            ;;
        _artifacts/*)
            rel=${source#_artifacts/}
            printf '%s/%s\n' "$REMOTE_ARTIFACTS_DIR" "$rel"
            ;;
        *)
            base=${source_path##*/}
            printf '%s/baselines/%s\n' "$REMOTE_ARTIFACTS_DIR" "$base"
            ;;
    esac
}

validate_declared_baselines() {
    [ -n "$TRAIN_COMMAND" ] || [ -n "$ROWS_MANIFEST" ] || return 0
    local baselines count i source source_path label expected latest actual
    baselines=$(declared_baselines_json)
    count=$(printf '%s\n' "$baselines" | jq 'length')
    [ "$count" -gt 0 ] || return 0
    i=0
    while [ "$i" -lt "$count" ]; do
        source=$(printf '%s\n' "$baselines" | jq -r ".[$i].path")
        source_path=$(baseline_source_path "$source")
        label=$(printf '%s\n' "$baselines" | jq -r ".[$i].label")
        expected=$(printf '%s\n' "$baselines" | jq -r ".[$i].completed_batch")
        [ -e "$source_path" ] ||
            die "baseline preflight failed: source checkpoint not found for $label: $source_path"
        latest="$source_path/latest.json"
        [ -f "$latest" ] ||
            die "baseline preflight failed: custody latest.json not found for $label: $latest"
        [ -n "$expected" ] && [ "$expected" != "null" ] ||
            die "baseline preflight failed: declared completed_batch missing for $label"
        actual=$(latest_pointer_completed_batches "$latest")
        [ -n "$actual" ] ||
            die "baseline preflight failed: completed_batch missing from $latest for $label"
        [ "$actual" = "$expected" ] ||
            die "baseline preflight failed: completed_batch mismatch for $label: declared $expected but latest.json has $actual"
        i=$((i + 1))
    done
}

sync_declared_baselines() {
    local baselines count i source source_path target
    baselines=$(declared_baselines_json)
    count=$(printf '%s\n' "$baselines" | jq 'length')
    [ "$count" -gt 0 ] || return 0
    i=0
    while [ "$i" -lt "$count" ]; do
        source=$(printf '%s\n' "$baselines" | jq -r ".[$i].path")
        source_path=$(baseline_source_path "$source")
        target=$(baseline_remote_target "$source")
        log "staging declared baseline $source_path -> $target"
        provider_exec "mkdir -p $(sq "$(dirname "$target")")"
        if [ -d "$source_path" ]; then
            provider_copy "$source_path/" "$target/" 1
        else
            provider_copy "$source_path" "$target" 0
        fi
        i=$((i + 1))
    done
}

validate_remote_declared_baselines() {
    local baselines count i source target label expected latest
    baselines=$(declared_baselines_json)
    count=$(printf '%s\n' "$baselines" | jq 'length')
    [ "$count" -gt 0 ] || return 0
    i=0
    while [ "$i" -lt "$count" ]; do
        source=$(printf '%s\n' "$baselines" | jq -r ".[$i].path")
        target=$(baseline_remote_target "$source")
        label=$(printf '%s\n' "$baselines" | jq -r ".[$i].label")
        expected=$(printf '%s\n' "$baselines" | jq -r ".[$i].completed_batch")
        latest="$target/latest.json"
        provider_exec "test -e $(sq "$target") && test -f $(sq "$latest") && actual=\$(jq -r '.completed_training_batches // .completed_batches // .completed_batch // .completedBatch // .n_batches // .batch // .metadata.completed_training_batches // .metadata.completed_batches // .metadata.completed_batch // .metadata.completedBatch // .completed_coordinate.global_step // empty' $(sq "$latest")) && test -n \"\$actual\" && test \"\$actual\" = $(sq "$expected")" ||
            die "baseline preflight failed: remote staged baseline mismatch for $label at $latest (expected completed batch $expected)"
        i=$((i + 1))
    done
}

write_run_config() {
    local parent
    [ -n "$RUNPOD_RUN_CONFIG_FILE" ] || return 0
    parent=$(dirname "$RUNPOD_RUN_CONFIG_FILE")
    mkdir -p "$parent"
    jq -n \
        --arg pod_id "${POD_ID:-}" \
        --arg remote_run_dir "$REMOTE_RUN_DIR" \
        --arg remote_sentinel_dir "$REMOTE_SENTINEL_DIR" \
        --arg remote_checkpoint_dir "$REMOTE_RUN_DIR" \
        --arg remote_log_dir "$REMOTE_RUN_DIR/logs" \
        --arg remote_artifacts_dir "$REMOTE_ARTIFACTS_DIR" \
        --argjson baselines "$(declared_baselines_json)" \
        '{
          schema_version: 1,
          pod_id: $pod_id,
          remote_run_dir: $remote_run_dir,
          remote_sentinel_dir: $remote_sentinel_dir,
          remote_checkpoint_dir: $remote_checkpoint_dir,
          remote_log_dir: $remote_log_dir,
          remote_artifacts_dir: $remote_artifacts_dir,
          baselines: $baselines
        }' >"$RUNPOD_RUN_CONFIG_FILE"
}

sync_run_config() {
    [ -n "$RUNPOD_RUN_CONFIG_FILE" ] || return 0
    [ -f "$RUNPOD_RUN_CONFIG_FILE" ] || return 0
    provider_exec "mkdir -p $(sq "$REMOTE_RUN_DIR")"
    provider_copy "$RUNPOD_RUN_CONFIG_FILE" "$REMOTE_RUN_DIR/run-config.json" 0
    provider_exec "mkdir -p $(sq "$(dirname "$REMOTE_DEPLOY_CONFIG_PATH")")"
    provider_copy "$RUNPOD_RUN_CONFIG_FILE" "$REMOTE_DEPLOY_CONFIG_PATH" 0
}

sync_train_spec() {
    [ -n "$TRAIN_SPEC" ] || return 0
    provider_exec "mkdir -p $(sq "$REMOTE_RUN_DIR")"
    provider_copy "$TRAIN_SPEC" "$REMOTE_RUN_DIR/train-spec.json" 0
}

sync_rows_manifest() {
    [ -n "$ROWS_MANIFEST" ] || return 0
    provider_exec "mkdir -p $(sq "$REMOTE_RUN_DIR")"
    provider_copy "$ROWS_MANIFEST" "$REMOTE_RUN_DIR/rows-manifest.json" 0
}

remote_nohup_sentinel() {
    local label=$1
    local workdir=$2
    local command=$3
    local done_file=$4
    local failed_file=$5
    local log_file=$6
    local remote sentinel_command
    sentinel_command="cd $(sq "$workdir") && success=0; child=; mark_failed() { rc=\$?; if [ -n \"\$child\" ]; then kill \"\$child\" 2>/dev/null || true; fi; if [ \"\$success\" -ne 1 ]; then touch $(sq "$failed_file"); fi; exit \"\$rc\"; }; signal_failed() { rc=\$1; if [ -n \"\$child\" ]; then kill \"\$child\" 2>/dev/null || true; fi; touch $(sq "$failed_file"); exit \"\$rc\"; }; trap mark_failed EXIT; trap 'signal_failed 130' INT; trap 'signal_failed 143' TERM; trap 'signal_failed 129' HUP; { $command; } & child=\$!; wait \"\$child\"; rc=\$?; child=; if [ \"\$rc\" -eq 0 ]; then success=1; touch $(sq "$done_file"); else touch $(sq "$failed_file"); exit \"\$rc\"; fi"
    remote="mkdir -p $(sq "$REMOTE_SENTINEL_DIR") $(sq "$REMOTE_RUN_DIR/logs") && rm -f $(sq "$done_file") $(sq "$failed_file") && nohup bash -lc $(sq "$sentinel_command") >$(sq "$log_file") 2>&1 &"
    log "starting $label"
    provider_exec "$remote"
}

wait_for_sentinel() {
    local label=$1
    local done_file=$2
    local failed_file=$3
    if [ "$DRY_RUN" -eq 1 ]; then
        provider_exec "test -f $(sq "$done_file") || test -f $(sq "$failed_file")"
        return 0
    fi

    local deadline
    deadline=$((SECONDS + SENTINEL_TIMEOUT_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if provider_capture "test -f $(sq "$done_file")"; then
            log "$label complete"
            return 0
        fi
        if provider_capture "test -f $(sq "$failed_file")"; then
            die "$label failed; inspect $REMOTE_RUN_DIR/logs"
        fi
        sleep "$SENTINEL_POLL_SECONDS"
    done
    die "timed out waiting for $label sentinel"
}

wait_for_sentinel_result() {
    local label=$1
    local done_file=$2
    local failed_file=$3
    if [ "$DRY_RUN" -eq 1 ]; then
        provider_exec "test -f $(sq "$done_file") || test -f $(sq "$failed_file")"
        return 0
    fi

    local deadline
    deadline=$((SECONDS + SENTINEL_TIMEOUT_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if provider_capture "test -f $(sq "$done_file")"; then
            log "$label complete"
            return 0
        fi
        if provider_capture "test -f $(sq "$failed_file")"; then
            log "$label failed"
            return 1
        fi
        sleep "$SENTINEL_POLL_SECONDS"
    done
    die "timed out waiting for $label sentinel"
}

launch_row() {
    local row_id=$1 workdir=$2 command=$3
    local done_file failed_file log_file pid_file started_file remote cache_dir
    cache_dir="${JAX_COMPILATION_CACHE_DIR:-${RUNPOD_VOLUME_MOUNT:-/workspace}/jax_cache}"
    validate_row_id "$row_id" >/dev/null ||
        die "unsafe row id: $row_id (must match [A-Za-z0-9_.-]+)"
    done_file="$REMOTE_SENTINEL_DIR/${row_id}.done"
    failed_file="$REMOTE_SENTINEL_DIR/${row_id}.failed"
    log_file="$REMOTE_RUN_DIR/logs/${row_id}.log"
    pid_file="$REMOTE_SENTINEL_DIR/${row_id}.pid"
    started_file="$REMOTE_SENTINEL_DIR/${row_id}.started"
    remote="mkdir -p $(sq "$REMOTE_SENTINEL_DIR") $(sq "$REMOTE_RUN_DIR/logs") $(sq "$cache_dir") && rm -f $(sq "$done_file") $(sq "$failed_file") && touch $(sq "$started_file") && nohup bash -lc $(sq "cd $(sq "$workdir") && success=0; child=; mark_failed() { rc=\$?; if [ -n \"\$child\" ]; then kill \"\$child\" 2>/dev/null || true; fi; if [ \"\$success\" -ne 1 ]; then touch $(sq "$failed_file"); fi; exit \"\$rc\"; }; signal_failed() { rc=\$1; if [ -n \"\$child\" ]; then kill \"\$child\" 2>/dev/null || true; fi; touch $(sq "$failed_file"); exit \"\$rc\"; }; trap mark_failed EXIT; trap 'signal_failed 130' INT; trap 'signal_failed 143' TERM; trap 'signal_failed 129' HUP; echo \$\$ > $(sq "$pid_file") && export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=$(sq "$cache_dir") && ( $command ) & child=\$!; wait \"\$child\"; rc=\$?; child=; if [ \"\$rc\" -eq 0 ]; then success=1; touch $(sq "$done_file"); else touch $(sq "$failed_file"); exit \$rc; fi") >$(sq "$log_file") 2>&1 &"
    log "launching row $row_id"
    provider_exec "$remote"
}

wait_for_row_active_training() {
    local row_id=$1
    local done_file failed_file log_file timeout poll_seconds ready_probe ready_description
    local ready_regex ready_string
    timeout="${SENTINEL_TIMEOUT_SECONDS:-7200}"
    poll_seconds="${SENTINEL_POLL_SECONDS:-30}"
    ready_regex="${WARM_COMPILE_READY_REGEX:-([Bb]atch|[Ss]tep|[Ii]ter|[Ii]t)[[:space:]=:]+[0-9]+}"
    ready_string="${WARM_COMPILE_READY_STRING:-}"
    done_file="$REMOTE_SENTINEL_DIR/${row_id}.done"
    failed_file="$REMOTE_SENTINEL_DIR/${row_id}.failed"
    log_file="$REMOTE_RUN_DIR/logs/${row_id}.log"
    if [ -n "$ready_string" ]; then
        ready_probe="test -f $(sq "$log_file") && grep -F -- $(sq "$ready_string") $(sq "$log_file") >/dev/null 2>&1"
        ready_description="string $(sq "$ready_string")"
    elif [ -n "$ready_regex" ]; then
        ready_probe="test -f $(sq "$log_file") && grep -E -- $(sq "$ready_regex") $(sq "$log_file") >/dev/null 2>&1"
        ready_description="regex $(sq "$ready_regex")"
    else
        die "warm compile readiness requires WARM_COMPILE_READY_REGEX or WARM_COMPILE_READY_STRING"
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        log "dry-run: warm compile first would poll row $row_id log for active training ($ready_description)"
        provider_exec "$ready_probe || test -f $(sq "$done_file") || test -f $(sq "$failed_file")"
        return 0
    fi

    local deadline
    deadline=$((SECONDS + timeout))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if provider_capture "test -f $(sq "$failed_file")"; then
            die "warm compile row $row_id failed; inspect $REMOTE_RUN_DIR/logs/${row_id}.log"
        fi
        if provider_capture "$ready_probe"; then
            log "warm compile row $row_id reached active training; launching remaining rows"
            return 0
        fi
        if provider_capture "test -f $(sq "$done_file")"; then
            log "warm compile row $row_id completed before readiness marker matched; treating successful completion as warmed"
            return 0
        fi
        sleep "$poll_seconds"
    done
    die "timed out waiting for warm compile row $row_id log readiness marker ($ready_description)"
}

count_running_rows() {
    [ "$DRY_RUN" -eq 1 ] && { printf '0\n'; return 0; }
    provider_capture "n=0; for s in $(sq "$REMOTE_SENTINEL_DIR")/*.started; do [ -e \"\$s\" ] || continue; base=\${s%.started}; if [ ! -f \"\$base.done\" ] && [ ! -f \"\$base.failed\" ]; then n=\$((n+1)); fi; done; printf '%s' \"\$n\"" 2>/dev/null || printf '0'
}

launch_training() {
    local ids workdir command default_wd warm_compile_first
    default_wd=$(provider_workdir)
    warm_compile_first="${WARM_COMPILE_FIRST:-1}"

    if [ -n "$ROWS_MANIFEST" ]; then
        [ -f "$ROWS_MANIFEST" ] || die "rows manifest not found: $ROWS_MANIFEST"
        local validation
        validation=$(validate_rows_manifest <"$ROWS_MANIFEST") ||
            die "invalid rows manifest: $validation"
        log "rows manifest $validation (stagger=${ROW_LAUNCH_STAGGER_SECONDS}s cap=${MAX_PARALLEL_ROWS} warm_compile_first=${warm_compile_first})"
        sync_rows_manifest
        read_lines_into ids < <(rows_manifest_ids <"$ROWS_MANIFEST")
        local first=1 running row_id
        for row_id in "${ids[@]}"; do
            workdir=$(rows_manifest_field "$row_id" workdir <"$ROWS_MANIFEST")
            [ -n "$workdir" ] || workdir=$default_wd
            command=$(rows_manifest_field "$row_id" command <"$ROWS_MANIFEST")
            if [ "$DRY_RUN" -eq 0 ] && [ "$MAX_PARALLEL_ROWS" -gt 0 ]; then
                while :; do
                    running=$(count_running_rows)
                    [ "${running:-0}" -lt "$MAX_PARALLEL_ROWS" ] && break
                    log "row cap reached ($running/$MAX_PARALLEL_ROWS in flight); waiting"
                    sleep "$SENTINEL_POLL_SECONDS"
                done
            fi
            if [ "$first" -eq 0 ] && [ "$ROW_LAUNCH_STAGGER_SECONDS" -gt 0 ]; then
                [ "$DRY_RUN" -eq 0 ] && sleep "$ROW_LAUNCH_STAGGER_SECONDS"
            fi
            first=0
            launch_row "$row_id" "$workdir" "$command"
            if [ "$warm_compile_first" -eq 1 ]; then
                wait_for_row_active_training "$row_id"
                warm_compile_first=0
            fi
        done
        log "launched ${#ids[@]} row(s)"
        return 0
    fi

    [ -n "$TRAIN_COMMAND" ] || return 0
    launch_row "training" "$default_wd" "$TRAIN_COMMAND"
}
