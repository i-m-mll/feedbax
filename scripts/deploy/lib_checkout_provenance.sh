#!/usr/bin/env bash

# Best-effort checkout provenance for deploy and poll logs. Missing repositories
# or pin files are reported as unknown and never block the operational command.

checkout_identity() {
    local root=$1 sha=unknown branch=unknown dirty=unknown
    if git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        sha=$(git -C "$root" rev-parse HEAD 2>/dev/null || printf 'unknown')
        branch=$(git -C "$root" symbolic-ref --quiet --short HEAD 2>/dev/null ||
            printf 'detached')
        if [ -n "$(git -C "$root" status --porcelain 2>/dev/null)" ]; then
            dirty=true
        else
            dirty=false
        fi
    fi
    printf '%s %s %s\n' "$sha" "$branch" "$dirty"
}

consumer_feedbax_pin() {
    local consumer_root=$1 pin_file pin
    if [ -n "${FEEDBAX_PIN_SHA:-}" ]; then
        printf '%s\n' "$FEEDBAX_PIN_SHA"
        return 0
    fi
    pin_file="${FEEDBAX_PIN_FILE:-$consumer_root/ci/feedbax-ref.toml}"
    [ -f "$pin_file" ] || return 1
    pin=$(sed -nE 's/^[[:space:]]*rev[[:space:]]*=[[:space:]]*"([0-9a-fA-F]+)".*/\1/p' \
        "$pin_file" | head -1)
    [ -n "$pin" ] || return 1
    printf '%s\n' "$pin"
}

emit_checkout_provenance() {
    local script_path=$1 feedbax_root=$2 consumer_root=${3:-}
    local feedbax_sha feedbax_branch feedbax_dirty
    local consumer_sha=unknown consumer_branch=unknown consumer_dirty=unknown pin=unknown
    read -r feedbax_sha feedbax_branch feedbax_dirty < <(checkout_identity "$feedbax_root")
    if [ -n "$consumer_root" ]; then
        read -r consumer_sha consumer_branch consumer_dirty < <(
            checkout_identity "$consumer_root"
        )
        pin=$(consumer_feedbax_pin "$consumer_root" 2>/dev/null || printf 'unknown')
    fi
    printf 'provenance feedbax_sha=%s feedbax_branch=%s feedbax_dirty=%s consumer_sha=%s consumer_branch=%s consumer_dirty=%s feedbax_pin=%s script=%s\n' \
        "$feedbax_sha" "$feedbax_branch" "$feedbax_dirty" \
        "$consumer_sha" "$consumer_branch" "$consumer_dirty" "$pin" "$script_path" >&2
    if [ "$pin" != unknown ] && [ "$feedbax_sha" != unknown ] &&
        [[ "$feedbax_sha" != "$pin"* ]] && [[ "$pin" != "$feedbax_sha"* ]]; then
        printf 'WARNING: feedbax checkout/pin skew executing_sha=%s consumer_pin=%s (non-fatal)\n' \
            "$feedbax_sha" "$pin" >&2
    fi
}
