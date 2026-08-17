#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0

usage() {
    printf 'Usage: %s [--apply]\n' "$0"
    printf 'Default: dry-run. --apply deletes only cache files and approved empty directories.\n'
}

case "${1:-}" in
    "") ;;
    --apply) APPLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

remove_file() {
    local path="$1"
    [[ -e "$path" || -L "$path" ]] || return 0
    if [[ "$APPLY" -eq 1 ]]; then
        printf '[DELETE FILE] %s\n' "$path"
        rm -f -- "$path"
    else
        printf '[DRY-RUN DELETE FILE] %s\n' "$path"
    fi
}

remove_dir_if_empty() {
    local path="$1"
    [[ -d "$path" ]] || return 0
    if [[ "$APPLY" -eq 1 ]]; then
        printf '[RMDIR IF EMPTY] %s\n' "$path"
        rmdir -- "$path" 2>/dev/null || true
    else
        printf '[DRY-RUN RMDIR IF EMPTY] %s\n' "$path"
    fi
}

if [[ "$APPLY" -eq 0 ]]; then
    printf '[DRY-RUN] No files will be deleted. Use --apply only after review.\n'
fi

# Cache directory contents are removed one file at a time. No recursive rm is
# used. Directories are removed bottom-up only when empty.
while IFS= read -r -d '' cache_dir; do
    [[ -d "$cache_dir" ]] || continue
    while IFS= read -r -d '' file; do
        remove_file "$file"
    done < <(find "$cache_dir" -type f -print0)
    while IFS= read -r -d '' dir; do
        remove_dir_if_empty "$dir"
    done < <(find "$cache_dir" -depth -type d -print0)
done < <(find "$ROOT" -xdev -type d \( -name __pycache__ -o -name .pytest_cache \) -print0)

# Handle stray bytecode outside conventional cache directories.
while IFS= read -r -d '' file; do
    remove_file "$file"
done < <(find "$ROOT" -xdev -type f -name '*.pyc' \
    ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' -print0)

# Only unmistakable zero-byte temporary names are eligible. Empty Python
# package markers, YAML, logs, reports, checkpoints, results, and source files
# are deliberately excluded.
while IFS= read -r -d '' file; do
    remove_file "$file"
done < <(find "$ROOT" -xdev -type f -empty \
    \( -name '*.tmp' -o -name '*.temp' -o -name '*.swp' -o -name '*.swo' \
       -o -name '.DS_Store' -o -name 'nohup.out' \) -print0)

# Remove only empty generated directories. Never touch Git metadata,
# checkpoints, logs, reports, YAMLs, source trees, or tracking-result trees.
for generated_root in "$ROOT/output" "$ROOT/tensorboard"; do
    [[ -d "$generated_root" ]] || continue
    while IFS= read -r -d '' dir; do
        case "$dir" in
            *'/checkpoints/'*|*'/logs/'*|*'/tracking_results/'*|*'/final_paper_results/'*|*'/baseline_eval/'*|*'/cleanup/'*)
                continue
                ;;
        esac
        remove_dir_if_empty "$dir"
    done < <(find "$generated_root" -depth -type d -empty -print0)
done

printf 'Done. No checkpoint, YAML, log, test result, report, or source path is targeted.\n'
