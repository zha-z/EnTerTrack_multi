#!/usr/bin/env python3
"""Read-only Three-MDOT train/val natural visibility audit."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


VIEW_SUFFIX = {"1": "A", "2": "B", "3": "C"}
VIEWS = ("A", "B", "C")
PATTERNS = tuple(format(value, "03b") for value in range(8))


def read_nonempty_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_digest(dataset_root, sequence_names):
    digest = hashlib.sha256()
    file_count = 0
    for sequence_name in sorted(sequence_names):
        target, _ = sequence_name.rsplit("-", 1)
        for filename in ("groundtruth.txt", "occlusion.txt", "out_of_view.txt"):
            path = dataset_root / target / sequence_name / filename
            relative = path.relative_to(dataset_root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
            file_count += 1
    return digest.hexdigest(), file_count


def read_binary(path):
    values = [int(item) for item in read_nonempty_lines(path)]
    unique = set(values)
    if not unique.issubset({0, 1}):
        raise ValueError("non-binary annotation in {}: {}".format(path, sorted(unique)))
    return values


def read_groundtruth(path):
    boxes = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 4:
                raise ValueError("groundtruth row has fewer than 4 columns: {}:{}".format(path, row_index))
            box = [float(value) for value in row[:4]]
            if not all(math.isfinite(value) for value in box):
                raise ValueError("non-finite groundtruth row: {}:{}".format(path, row_index))
            boxes.append(box)
    return boxes


def load_view(dataset_root, sequence_name):
    target, _ = sequence_name.rsplit("-", 1)
    sequence_dir = dataset_root / target / sequence_name
    boxes = read_groundtruth(sequence_dir / "groundtruth.txt")
    occlusion = read_binary(sequence_dir / "occlusion.txt")
    out_of_view = read_binary(sequence_dir / "out_of_view.txt")
    lengths = {len(boxes), len(occlusion), len(out_of_view)}
    if len(lengths) != 1:
        raise ValueError(
            "annotation length mismatch for {}: gt={}, occlusion={}, out_of_view={}".format(
                sequence_name, len(boxes), len(occlusion), len(out_of_view)))
    valid = [box[2] > 0 and box[3] > 0 for box in boxes]
    visible = [
        (occlusion[index] == 0 and out_of_view[index] == 0 and valid[index])
        for index in range(len(boxes))
    ]
    return {
        "sequence": sequence_name,
        "boxes": boxes,
        "valid": valid,
        "occlusion": occlusion,
        "out_of_view": out_of_view,
        "visible": visible,
    }


def group_split(split_path):
    sequence_names = read_nonempty_lines(split_path)
    if len(sequence_names) != len(set(sequence_names)):
        raise ValueError("duplicate sequence in {}".format(split_path))
    groups = defaultdict(dict)
    for sequence_name in sequence_names:
        target, suffix = sequence_name.rsplit("-", 1)
        if suffix not in VIEW_SUFFIX:
            raise ValueError("unsupported view suffix: {}".format(sequence_name))
        view = VIEW_SUFFIX[suffix]
        if view in groups[target]:
            raise ValueError("duplicate target/view: {} {}".format(target, view))
        groups[target][view] = sequence_name
    incomplete = {target: sorted(group) for target, group in groups.items() if set(group) != set(VIEWS)}
    if incomplete:
        raise ValueError("incomplete target groups: {}".format(incomplete))
    return sequence_names, dict(sorted(groups.items()))


def pct(numerator, denominator):
    if denominator == 0:
        return ""
    return "{:.6f}".format(100.0 * numerator / denominator)


def ratio(numerator, denominator):
    if denominator == 0:
        return ""
    return "{:.9f}".format(numerator / denominator)


def percentile_linear(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def find_runs(flags):
    runs = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def run_row(scope, target, receiver, lengths):
    return {
        "scope": scope,
        "target": target,
        "receiver": receiver,
        "run_count": len(lengths),
        "asymmetric_frame_count": sum(lengths),
        "mean_run_length": "{:.6f}".format(sum(lengths) / len(lengths)) if lengths else "0.000000",
        "median_run_length": "{:.6f}".format(statistics.median(lengths)) if lengths else "0.000000",
        "p90_run_length": "{:.6f}".format(percentile_linear(lengths, 0.90)),
        "max_run_length": max(lengths) if lengths else 0,
    }


def empty_receiver_counts():
    return {
        "receiver_frames": 0,
        "R2_two_visible_senders": 0,
        "R1_one_visible_sender": 0,
        "R0_zero_visible_senders": 0,
        "natural_asymmetric": 0,
        "any_sender_occlusion": 0,
        "any_sender_out_of_view": 0,
        "any_sender_invalid_bbox": 0,
    }


def add_counts(destination, source):
    for key in destination:
        destination[key] += source[key]


def receiver_summary_row(split, receiver, counts):
    total = counts["receiver_frames"]
    return {
        "split": split,
        "receiver": receiver,
        "receiver_frame_count": total,
        "R2_two_visible_senders_count": counts["R2_two_visible_senders"],
        "R2_two_visible_senders_percentage": pct(counts["R2_two_visible_senders"], total),
        "R1_one_visible_sender_count": counts["R1_one_visible_sender"],
        "R1_one_visible_sender_percentage": pct(counts["R1_one_visible_sender"], total),
        "R0_zero_visible_senders_count": counts["R0_zero_visible_senders"],
        "R0_zero_visible_senders_percentage": pct(counts["R0_zero_visible_senders"], total),
        "natural_asymmetric_count": counts["natural_asymmetric"],
        "natural_asymmetric_percentage": pct(counts["natural_asymmetric"], total),
        "any_sender_occlusion_count": counts["any_sender_occlusion"],
        "any_sender_occlusion_percentage": pct(counts["any_sender_occlusion"], total),
        "any_sender_out_of_view_count": counts["any_sender_out_of_view"],
        "any_sender_out_of_view_percentage": pct(counts["any_sender_out_of_view"], total),
        "any_sender_invalid_bbox_count": counts["any_sender_invalid_bbox"],
        "any_sender_invalid_bbox_percentage": pct(counts["any_sender_invalid_bbox"], total),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_split(split, split_path, dataset_root, output_dir):
    sequence_names, groups = group_split(split_path)
    pattern_counts = Counter({pattern: 0 for pattern in PATTERNS})
    receiver_totals = {view: empty_receiver_counts() for view in VIEWS}
    run_lengths_by_receiver = {view: [] for view in VIEWS}
    run_rows = []
    per_target_rows = []
    total_sync_frames = 0
    total_discarded_tail = 0
    invalid_bbox_count = 0
    observed_occlusion = set()
    observed_out_of_view = set()

    for target, view_names in groups.items():
        data = {view: load_view(dataset_root, view_names[view]) for view in VIEWS}
        lengths = {view: len(data[view]["visible"]) for view in VIEWS}
        sync_len = min(lengths.values())
        discarded_tail = sum(lengths[view] - sync_len for view in VIEWS)
        total_sync_frames += sync_len
        total_discarded_tail += discarded_tail
        target_receiver = {view: empty_receiver_counts() for view in VIEWS}
        target_pattern_counts = Counter({pattern: 0 for pattern in PATTERNS})
        target_asymmetric_flags = {view: [] for view in VIEWS}

        for view in VIEWS:
            observed_occlusion.update(data[view]["occlusion"])
            observed_out_of_view.update(data[view]["out_of_view"])
            invalid_bbox_count += sum(not value for value in data[view]["valid"][:sync_len])

        for frame_index in range(sync_len):
            visible = {view: bool(data[view]["visible"][frame_index]) for view in VIEWS}
            pattern = "".join("1" if visible[view] else "0" for view in VIEWS)
            pattern_counts[pattern] += 1
            target_pattern_counts[pattern] += 1

            for receiver in VIEWS:
                senders = [view for view in VIEWS if view != receiver]
                asymmetric = visible[receiver] and not all(visible[sender] for sender in senders)
                target_asymmetric_flags[receiver].append(asymmetric)
                if not visible[receiver]:
                    continue
                counts = target_receiver[receiver]
                counts["receiver_frames"] += 1
                visible_senders = sum(visible[sender] for sender in senders)
                if visible_senders == 2:
                    counts["R2_two_visible_senders"] += 1
                elif visible_senders == 1:
                    counts["R1_one_visible_sender"] += 1
                    counts["natural_asymmetric"] += 1
                else:
                    counts["R0_zero_visible_senders"] += 1
                    counts["natural_asymmetric"] += 1
                if any(data[sender]["occlusion"][frame_index] == 1 for sender in senders):
                    counts["any_sender_occlusion"] += 1
                if any(data[sender]["out_of_view"][frame_index] == 1 for sender in senders):
                    counts["any_sender_out_of_view"] += 1
                if any(not data[sender]["valid"][frame_index] for sender in senders):
                    counts["any_sender_invalid_bbox"] += 1

        for receiver in VIEWS:
            add_counts(receiver_totals[receiver], target_receiver[receiver])
            lengths_for_target = find_runs(target_asymmetric_flags[receiver])
            run_lengths_by_receiver[receiver].extend(lengths_for_target)
            run_rows.append(run_row("target_receiver", target, receiver, lengths_for_target))

        receiver_total = sum(target_receiver[view]["receiver_frames"] for view in VIEWS)
        asymmetric_total = sum(target_receiver[view]["natural_asymmetric"] for view in VIEWS)
        row = {
            "split": split,
            "target": target,
            "A_annotation_frames": lengths["A"],
            "B_annotation_frames": lengths["B"],
            "C_annotation_frames": lengths["C"],
            "synchronized_triplets": sync_len,
            "discarded_unsynchronized_tail_frames": discarded_tail,
            "common_visible_triplets": target_pattern_counts["111"],
            "A_receiver_visible_frames": target_receiver["A"]["receiver_frames"],
            "B_receiver_visible_frames": target_receiver["B"]["receiver_frames"],
            "C_receiver_visible_frames": target_receiver["C"]["receiver_frames"],
            "receiver_visible_total": receiver_total,
            "A_natural_asymmetric_frames": target_receiver["A"]["natural_asymmetric"],
            "A_natural_asymmetric_percentage": pct(target_receiver["A"]["natural_asymmetric"], target_receiver["A"]["receiver_frames"]),
            "B_natural_asymmetric_frames": target_receiver["B"]["natural_asymmetric"],
            "B_natural_asymmetric_percentage": pct(target_receiver["B"]["natural_asymmetric"], target_receiver["B"]["receiver_frames"]),
            "C_natural_asymmetric_frames": target_receiver["C"]["natural_asymmetric"],
            "C_natural_asymmetric_percentage": pct(target_receiver["C"]["natural_asymmetric"], target_receiver["C"]["receiver_frames"]),
            "natural_asymmetric_receiver_frames": asymmetric_total,
            "natural_asymmetric_percentage": pct(asymmetric_total, receiver_total),
        }
        per_target_rows.append(row)

    overall_receiver = empty_receiver_counts()
    receiver_summary_rows = []
    for receiver in VIEWS:
        receiver_summary_rows.append(receiver_summary_row(split, receiver, receiver_totals[receiver]))
        add_counts(overall_receiver, receiver_totals[receiver])
        run_rows.append(run_row("receiver_overall", "", receiver, run_lengths_by_receiver[receiver]))
    receiver_summary_rows.append(receiver_summary_row(split, "overall", overall_receiver))
    pooled_runs = []
    for receiver in VIEWS:
        pooled_runs.extend(run_lengths_by_receiver[receiver])
    run_rows.append(run_row("split_overall", "", "overall", pooled_runs))

    triplet_rows = []
    for pattern in PATTERNS:
        count = pattern_counts[pattern]
        triplet_rows.append({
            "split": split,
            "pattern_ABC": pattern,
            "visible_view_count": pattern.count("1"),
            "frame_count": count,
            "percentage": pct(count, total_sync_frames),
        })

    level_rows = []
    for visible_views in (3, 2, 1, 0):
        count = sum(pattern_counts[pattern] for pattern in PATTERNS if pattern.count("1") == visible_views)
        level_rows.append({
            "split": split,
            "visible_view_count": visible_views,
            "frame_count": count,
            "percentage": pct(count, total_sync_frames),
        })

    write_csv(output_dir / "{}_triplet_visibility_patterns.csv".format(split), triplet_rows, list(triplet_rows[0]))
    write_csv(output_dir / "{}_visibility_levels.csv".format(split), level_rows, list(level_rows[0]))
    write_csv(output_dir / "{}_receiver_visibility_summary.csv".format(split), receiver_summary_rows, list(receiver_summary_rows[0]))
    write_csv(output_dir / "{}_per_target_visibility.csv".format(split), per_target_rows, list(per_target_rows[0]))
    write_csv(output_dir / "{}_asymmetric_run_lengths.csv".format(split), run_rows, list(run_rows[0]))

    expected_receiver = sum(pattern_counts[pattern] * pattern.count("1") for pattern in PATTERNS)
    checks = {
        "pattern_sum_matches_sync": sum(pattern_counts.values()) == total_sync_frames,
        "level_sum_matches_sync": sum(int(row["frame_count"]) for row in level_rows) == total_sync_frames,
        "receiver_total_matches_patterns": overall_receiver["receiver_frames"] == expected_receiver,
        "receiver_state_partition": all(
            receiver_totals[view]["receiver_frames"] ==
            receiver_totals[view]["R2_two_visible_senders"] +
            receiver_totals[view]["R1_one_visible_sender"] +
            receiver_totals[view]["R0_zero_visible_senders"]
            for view in VIEWS),
        "run_frames_match_asymmetric": sum(pooled_runs) == overall_receiver["natural_asymmetric"],
        "binary_occlusion": observed_occlusion.issubset({0, 1}),
        "binary_out_of_view": observed_out_of_view.issubset({0, 1}),
    }
    if not all(checks.values()):
        raise RuntimeError("{} consistency checks failed: {}".format(split, checks))

    annotation_sha256, annotation_file_count = annotation_digest(dataset_root, sequence_names)
    return {
        "split": split,
        "split_path": split_path,
        "split_sha256": sha256_file(split_path),
        "sequence_names": sequence_names,
        "target_count": len(groups),
        "sequence_count": len(sequence_names),
        "synchronized_triplets": total_sync_frames,
        "discarded_unsynchronized_tail_frames": total_discarded_tail,
        "N_common": pattern_counts["111"],
        "N_receiver_A": receiver_totals["A"]["receiver_frames"],
        "N_receiver_B": receiver_totals["B"]["receiver_frames"],
        "N_receiver_C": receiver_totals["C"]["receiver_frames"],
        "N_receiver_total": overall_receiver["receiver_frames"],
        "natural_asymmetric_receiver_frames": overall_receiver["natural_asymmetric"],
        "natural_asymmetric_ratio_pct": pct(overall_receiver["natural_asymmetric"], overall_receiver["receiver_frames"]),
        "invalid_bbox_count_within_sync": invalid_bbox_count,
        "occlusion_values": sorted(observed_occlusion),
        "out_of_view_values": sorted(observed_out_of_view),
        "annotation_sha256": annotation_sha256,
        "annotation_file_count": annotation_file_count,
        "checks": checks,
        "per_target_rows": per_target_rows,
    }


def coverage_row(result):
    current_capacity = 3 * result["N_common"]
    extra = result["N_receiver_total"] - current_capacity
    common_per_target = [int(row["common_visible_triplets"]) for row in result["per_target_rows"]]
    asymmetric_per_target = [
        int(row["natural_asymmetric_receiver_frames"])
        for row in result["per_target_rows"]
    ]
    asymmetric_sorted = sorted(asymmetric_per_target, reverse=True)
    asymmetric_total = result["natural_asymmetric_receiver_frames"]
    return {
        "split": result["split"],
        "synchronized_triplets": result["synchronized_triplets"],
        "N_common": result["N_common"],
        "current_common_visible_receiver_capacity": current_capacity,
        "N_receiver_A": result["N_receiver_A"],
        "N_receiver_B": result["N_receiver_B"],
        "N_receiver_C": result["N_receiver_C"],
        "N_receiver_total": result["N_receiver_total"],
        "receiver_visible_expansion_ratio": ratio(result["N_receiver_total"], current_capacity),
        "extra_receiver_frames": extra,
        "percentage_increase": pct(extra, current_capacity),
        "natural_asymmetric_receiver_frames": result["natural_asymmetric_receiver_frames"],
        "natural_asymmetric_percentage": result["natural_asymmetric_ratio_pct"],
        "common_per_target_min": min(common_per_target),
        "common_per_target_max": max(common_per_target),
        "common_per_target_mean": "{:.6f}".format(sum(common_per_target) / len(common_per_target)),
        "common_per_target_median": "{:.6f}".format(statistics.median(common_per_target)),
        "targets_with_natural_asymmetry": sum(value > 0 for value in asymmetric_per_target),
        "top1_target_asymmetry_share_percentage": pct(sum(asymmetric_sorted[:1]), asymmetric_total),
        "top3_target_asymmetry_share_percentage": pct(sum(asymmetric_sorted[:3]), asymmetric_total),
    }


def git_output(repo_root, *args):
    return subprocess.check_output(["git"] + list(args), cwd=str(repo_root), text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/data2/Three-MDOT"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": repo_root / "lib/train/data_specs/threemdot/threemdot_train.txt",
        "val": repo_root / "lib/train/data_specs/threemdot/threemdot_val.txt",
    }
    results = {
        split: audit_split(split, split_path, args.dataset_root, output_dir)
        for split, split_path in split_paths.items()
    }
    train_targets = {row["target"] for row in results["train"]["per_target_rows"]}
    val_targets = {row["target"] for row in results["val"]["per_target_rows"]}
    if train_targets & val_targets:
        raise RuntimeError("train/val target overlap: {}".format(sorted(train_targets & val_targets)))

    coverage_rows = [coverage_row(results[split]) for split in ("train", "val")]
    write_csv(output_dir / "coverage_comparison.csv", coverage_rows, list(coverage_rows[0]))

    artifact_names = [
        "train_triplet_visibility_patterns.csv", "val_triplet_visibility_patterns.csv",
        "train_visibility_levels.csv", "val_visibility_levels.csv",
        "train_receiver_visibility_summary.csv", "val_receiver_visibility_summary.csv",
        "train_per_target_visibility.csv", "val_per_target_visibility.csv",
        "train_asymmetric_run_lengths.csv", "val_asymmetric_run_lengths.csv",
        "coverage_comparison.csv", "audit_visibility.py",
        "PROTOCOL_ZH.md", "REPORT_ZH.md", "COMMANDS_ZH.md",
        "visibility_schema_audit.md",
    ]
    artifacts = {
        name: {"sha256": sha256_file(output_dir / name)}
        for name in artifact_names
    }
    provenance = {
        "task": "D2-P0 Three-MDOT Natural Visibility and Receiver-visible Sampling Audit",
        "date": "2026-08-30",
        "branch": git_output(repo_root, "branch", "--show-current"),
        "source_head": git_output(repo_root, "rev-parse", "HEAD"),
        "dataset_root": str(args.dataset_root),
        "visibility_definition": "(occlusion == 0) AND (out_of_view == 0) AND (bbox_w > 0) AND (bbox_h > 0)",
        "synchronization_definition": "same target and 0-based frame index; length=min(A,B,C)",
        "percentage_unit": "percent_0_to_100",
        "p90_method": "linear interpolation at 0.90*(n-1)",
        "training_run": False,
        "validation_tracking_rollout_run": False,
        "official_test_run": False,
        "official_test_accessed": False,
        "model_or_training_code_modified": False,
        "train_val_target_overlap_count": 0,
        "splits": {},
        "artifacts": artifacts,
        "python": sys.version.split()[0],
        "command": "PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python docs/results/threemdot_receiver_visibility_audit_20260830/audit_visibility.py --output-dir docs/results/threemdot_receiver_visibility_audit_20260830",
    }
    for split in ("train", "val"):
        result = results[split]
        provenance["splits"][split] = {
            key: result[key]
            for key in (
                "target_count", "sequence_count", "synchronized_triplets",
                "discarded_unsynchronized_tail_frames", "N_common", "N_receiver_A",
                "N_receiver_B", "N_receiver_C", "N_receiver_total",
                "natural_asymmetric_receiver_frames", "natural_asymmetric_ratio_pct",
                "invalid_bbox_count_within_sync", "occlusion_values", "out_of_view_values",
                "split_sha256", "annotation_sha256", "annotation_file_count", "checks")
        }
        provenance["splits"][split]["split_file"] = str(result["split_path"].relative_to(repo_root))

    with (output_dir / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        split: coverage_row(results[split]) for split in ("train", "val")
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
