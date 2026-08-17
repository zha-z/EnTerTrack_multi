#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_ROOT="$ROOT/output/_archive_cleanup_20260704"
LOG_FILE="$ROOT/output/cleanup/archive_pcum_unused_files.log"
APPLY=0

usage() {
    printf 'Usage: %s [--apply]\n' "$0"
    printf 'Default: dry-run. --apply moves listed candidates into %s.\n' "$ARCHIVE_ROOT"
}

case "${1:-}" in
    "") ;;
    --apply) APPLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

declare -a CANDIDATES=()
declare -A SEEN=()

add_candidate() {
    local path="$1"
    if [[ -z "${SEEN[$path]:-}" ]]; then
        CANDIDATES+=("$path")
        SEEN[$path]=1
    fi
}

# High-confidence historical training outputs. None are used by the final
# no-GT result, baseline comparison, or paper tables.
for rel in \
    output/pcum_real_target_v1 \
    output/pcum_real_multiview_v1 \
    output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep5 \
    output/_short_pcum_supervision_e1_paired_lr8e6_ep5 \
    output/_short_pcum_supervision_e2a_safe_m0_lr8e6_ep5 \
    output/_short_pcum_supervision_e2b_safe_m0001_lr8e6_ep5 \
    output/_short_pcum_supervision_e3_safe_m0_lr4e5_ep5 \
    output/_short_pcum_supervision_e4_safe_m0_lr8e5_ep5 \
    output/pcum_ablation_group1 \
    output/pcum_ablation_group1_v2 \
    output/pcum_ablation_group3 \
    output/pcum_real_target_stable \
    output/pcum_real_target_stable_v2 \
    output/pcum_real_allviews_stable \
    output/pcum_real_allviews_stable_v2 \
    output/pcum_real_allviews_a_focus \
    output/pcum_long_occ \
    output/pcum_paper \
    output/pcum_supervision_pipeline \
    output/analysis/pcum_visual_smoke \
    tensorboard/train/entertrack/pcum_supervision_smoke
do
    add_candidate "$ROOT/$rel"
done

# Old pcum_ablation_current branches. Keep the independently trained baseline.
for name in \
    pcum_ablation_current_a_weight \
    pcum_ablation_current_allviews_equal \
    pcum_ablation_current_dropout \
    pcum_ablation_current_full \
    pcum_ablation_current_full_crosslayer \
    pcum_ablation_current_local_allviews \
    pcum_ablation_current_local_view0 \
    pcum_ablation_current_real_target
do
    add_candidate "$ROOT/output/pcum_ablation_current/checkpoints/train/entertrack/$name"
done

# Keep validation-selection epochs 5/10/15/20/25/30/35/40. Archive the
# remaining checkpoints from the final E4 and independent-baseline runs.
for base in \
    "$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40" \
    "$ROOT/output/pcum_ablation_current/checkpoints/train/entertrack/pcum_ablation_current_baseline"
do
    for epoch in $(seq 1 40); do
        case "$epoch" in
            5|10|15|20|25|30|35|40) continue ;;
        esac
        printf -v epoch4 '%04d' "$epoch"
        add_candidate "$base/EnTeRTrack_ep${epoch4}.pth.tar"
    done
done

# Generated short-run and obsolete sweep YAMLs. Canonical final no-GT,
# baseline, and validation-selection YAMLs are intentionally not included.
while IFS= read -r -d '' path; do
    add_candidate "$path"
done < <(find "$ROOT/experiments/entertrack" -maxdepth 1 -type f \
    \( -name '_short_*.yaml' -o -name '_eval__short_*.yaml' \) -print0)

for rel in \
    experiments/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0010_t0_pcum_disabled.yaml \
    experiments/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0010_t2_safe.yaml \
    experiments/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep5.yaml \
    experiments/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep5_t1_local_only.yaml \
    experiments/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep5_t2_raw.yaml \
    experiments/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep5_t2_safe.yaml \
    pcum_run301_302_check.txt \
    prompt.txt
do
    add_candidate "$ROOT/$rel"
done

# Historical result runids. Preserve final no-GT 12550-12556 and compatibility
# anchors 12152/12652. Validation and baseline runids 15xxx-17xxx are untouched.
RESULT_ROOT="$ROOT/output/test/tracking_results/entertrack"
if [[ -d "$RESULT_ROOT" ]]; then
    while IFS= read -r -d '' path; do
        name="${path##*/}"
        runid="${name##*_}"
        case "$runid" in
            12152|12550|12551|12552|12553|12554|12555|12556|12652)
                continue
                ;;
            92??|95??|96??|970?|12???|13???|14???)
                add_candidate "$path"
                ;;
        esac
    done < <(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)
fi

log_line() {
    local line="$1"
    printf '%s\n' "$line"
    if [[ "$APPLY" -eq 1 ]]; then
        printf '%s\n' "$line" >> "$LOG_FILE"
    fi
}

if [[ "$APPLY" -eq 1 ]]; then
    mkdir -p "$ARCHIVE_ROOT" "$(dirname "$LOG_FILE")"
    printf '\n[%s] archive apply start\n' "$(date '+%F %T')" >> "$LOG_FILE"
else
    printf '[DRY-RUN] No files will be moved. Use --apply after reviewing the inventory.\n'
fi

candidate_bytes=0
candidate_count=0
for source in "${CANDIDATES[@]}"; do
    relative="${source#"$ROOT/"}"
    destination="$ARCHIVE_ROOT/$relative"

    if [[ ! -e "$source" && ! -L "$source" ]]; then
        if [[ -e "$destination" || -L "$destination" ]]; then
            log_line "[ALREADY_ARCHIVED] $relative"
        else
            log_line "[MISSING] $relative"
        fi
        continue
    fi
    if [[ -e "$destination" || -L "$destination" ]]; then
        log_line "[SKIP_DEST_EXISTS] $relative"
        continue
    fi

    size=$(du -sb "$source" 2>/dev/null | awk '{print $1}')
    candidate_bytes=$((candidate_bytes + size))
    candidate_count=$((candidate_count + 1))

    if [[ "$APPLY" -eq 0 ]]; then
        log_line "[DRY-RUN] mv -- '$source' '$destination'"
        continue
    fi

    mkdir -p "$(dirname "$destination")"
    log_line "[MOVE] $source -> $destination"
    mv -- "$source" "$destination"
done

printf 'Candidates present: %d\n' "$candidate_count"
awk -v bytes="$candidate_bytes" 'BEGIN {printf "Candidate footprint: %.2f GiB\n", bytes/1024/1024/1024}'
printf 'Archive location is on the same filesystem; moving alone releases 0 bytes.\n'
if [[ "$APPLY" -eq 1 ]]; then
    printf 'Log: %s\n' "$LOG_FILE"
fi
