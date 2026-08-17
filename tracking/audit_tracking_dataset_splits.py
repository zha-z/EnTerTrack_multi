#!/usr/bin/env python3
"""Audit tracking dataset splits without constructing a tracker.

The default invocation audits the repository's Three-MDOT lists.  Additional
datasets can be supplied with repeatable ``--split-spec`` arguments using
``DATASET:SPLIT:LIST:ROOT``.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from dataset_development_utils import (
        assert_targets_not_split,
        group_by_target,
        locate_sequence,
        parse_sequence_name,
        read_numeric_rows,
        read_sequence_list,
        split_overlap_rows,
        write_csv,
    )
except ImportError:  # pragma: no cover - package import used by unit tests
    from tracking.dataset_development_utils import (
        assert_targets_not_split,
        group_by_target,
        locate_sequence,
        parse_sequence_name,
        read_numeric_rows,
        read_sequence_list,
        split_overlap_rows,
        write_csv,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = REPO_ROOT / "lib" / "train" / "data_specs" / "threemdot"
REQUIRED_SEQUENCE_FIELDS = [
    "dataset",
    "split",
    "target_id",
    "view_id",
    "sequence_name",
    "frame_count",
    "has_gt",
    "has_occlusion",
    "has_out_of_view",
    "used_in_training",
    "used_in_validation",
    "reserved_for_test",
    "eligible_for_dev",
    "eligibility_reason",
    "sequence_path",
]


def parse_split_spec(value: str) -> Tuple[str, str, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--split-spec must be DATASET:SPLIT:LIST:ROOT, got {!r}".format(value)
        )
    return parts[0], parts[1], Path(parts[2]), Path(parts[3])


def eligibility_for(split: str, used_in_training: bool) -> Tuple[bool, str]:
    normalized = split.lower()
    if normalized == "test":
        return False, "formal_test_reserved"
    if used_in_training:
        return False, "training_domain_diagnostic_only"
    if normalized == "val":
        return True, "existing_development_split_not_independent_after_reuse"
    if normalized in {"unassigned", "extra"}:
        return False, "requires_split_provenance_and_leakage_review"
    return False, "not_approved_for_development"


def build_sequence_rows(
    dataset: str,
    split: str,
    sequence_names: Sequence[str],
    root: Path,
    scan_annotations: bool,
) -> List[dict]:
    rows: List[dict] = []
    used_in_training = split.lower() == "train"
    eligible, reason = eligibility_for(split, used_in_training)
    for sequence_name in sequence_names:
        target_id, view_id = parse_sequence_name(sequence_name)
        sequence_path = locate_sequence(root, sequence_name)
        gt_path = sequence_path / "groundtruth.txt"
        occlusion_path = sequence_path / "occlusion.txt"
        out_of_view_path = sequence_path / "out_of_view.txt"
        frame_count = ""
        if scan_annotations:
            frame_count = len(read_numeric_rows(gt_path, expected_columns=4))
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "target_id": target_id,
                "view_id": view_id,
                "sequence_name": sequence_name,
                "frame_count": frame_count,
                "has_gt": gt_path.is_file(),
                "has_occlusion": occlusion_path.is_file(),
                "has_out_of_view": out_of_view_path.is_file(),
                "used_in_training": used_in_training,
                "used_in_validation": split.lower() == "val",
                "reserved_for_test": split.lower() == "test",
                "eligible_for_dev": eligible,
                "eligibility_reason": reason,
                "sequence_path": str(sequence_path),
            }
        )
    return rows


def target_inventory(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str, str], List[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["split"], row["target_id"])].append(row)
    result = []
    for (dataset, split, target_id), target_rows in sorted(grouped.items()):
        views = sorted(str(row["view_id"]) for row in target_rows)
        counts = [row["frame_count"] for row in target_rows if row["frame_count"] != ""]
        result.append(
            {
                "dataset": dataset,
                "split": split,
                "target_id": target_id,
                "view_count": len(views),
                "view_ids": ";".join(views),
                "all_views_grouped": True,
                "frame_count": sum(int(value) for value in counts) if counts else "",
                "used_in_training": any(row["used_in_training"] for row in target_rows),
                "reserved_for_test": any(row["reserved_for_test"] for row in target_rows),
                "eligible_for_dev": all(row["eligible_for_dev"] for row in target_rows),
                "eligibility_reason": target_rows[0]["eligibility_reason"],
            }
        )
    return result


def dataset_inventory(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["split"])].append(row)
    result = []
    for (dataset, split), split_rows in sorted(grouped.items()):
        counts = [row["frame_count"] for row in split_rows if row["frame_count"] != ""]
        result.append(
            {
                "dataset": dataset,
                "split": split,
                "target_count": len({row["target_id"] for row in split_rows}),
                "sequence_view_count": len(split_rows),
                "frame_count": sum(int(value) for value in counts) if counts else "",
                "all_have_gt": all(row["has_gt"] for row in split_rows),
                "all_have_occlusion": all(row["has_occlusion"] for row in split_rows),
                "all_have_out_of_view": all(row["has_out_of_view"] for row in split_rows),
            }
        )
    return result


def extract_training_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required to audit training YAML files.") from error
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Training config not found: {}".format(path))
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = payload.get("DATA", {})
    train = data.get("TRAIN", {})
    val = data.get("VAL", {})
    model = payload.get("MODEL", {})
    return {
        "config": path.stem,
        "path": str(path),
        "train_datasets": train.get("DATASETS_NAME", []),
        "train_ratios": train.get("DATASETS_RATIO", []),
        "train_samples_per_epoch": train.get("SAMPLE_PER_EPOCH", ""),
        "validation_datasets": val.get("DATASETS_NAME", []),
        "pretrain_file": model.get("PRETRAIN_FILE", ""),
    }


def render_training_report(config_rows: Sequence[dict]) -> str:
    lines = [
        "# Current training data audit",
        "",
        "This is a configuration-provenance audit; it does not claim that an unlogged external pretraining stage used no other data.",
        "",
        "| Config | Training datasets | Validation datasets | Pretrain/checkpoint |",
        "|---|---|---|---|",
    ]
    for row in config_rows:
        lines.append(
            "| {config} | `{train}` | `{val}` | `{pretrain}` |".format(
                config=row["config"],
                train=json.dumps(row["train_datasets"]),
                val=json.dumps(row["validation_datasets"]),
                pretrain=row["pretrain_file"],
            )
        )
    lines.extend(
        [
            "",
            "A0 is an inference configuration over the E4 epoch-15 checkpoint rather than a separate training run. D1 and D2 initialize from that E4 checkpoint and list only `THREEMDOT` as their fine-tuning dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_summary(
    inventory: Sequence[dict], overlap_rows: Sequence[dict], scan_annotations: bool
) -> str:
    lines = [
        "# Dataset audit summary",
        "",
        "Result label: **dataset/protocol audit**, not a tracking result.",
        "",
        "| Dataset | Split | Targets | Sequence-views | Frames |",
        "|---|---|---:|---:|---:|",
    ]
    for row in inventory:
        frames = row["frame_count"] if row["frame_count"] != "" else "pending annotation scan"
        lines.append(
            "| {dataset} | {split} | {target_count} | {sequence_view_count} | {frames} |".format(
                frames=frames, **row
            )
        )
    failures = [row for row in overlap_rows if row["status"] == "FAIL"]
    lines.extend(
        [
            "",
            "- Target grouping is the split unit; views must never be randomized independently.",
            "- Split overlap status: **{}**.".format("FAIL" if failures else "PASS"),
            "- Annotation scan: **{}**.".format("complete" if scan_annotations else "not run"),
            "- Exact-name/target overlap does not rule out copied media or same-scene leakage; content-level provenance remains a separate audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data2/Three-MDOT"))
    parser.add_argument("--train-list", type=Path, default=DEFAULT_SPLIT_DIR / "threemdot_train.txt")
    parser.add_argument("--val-list", type=Path, default=DEFAULT_SPLIT_DIR / "threemdot_val.txt")
    parser.add_argument("--test-list", type=Path, default=DEFAULT_SPLIT_DIR / "threemdot_test.txt")
    parser.add_argument("--split-spec", action="append", type=parse_split_spec, default=[])
    parser.add_argument("--training-config", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "dataset_development_audit")
    parser.add_argument("--scan-annotations", action="store_true", help="Read GT to count frames; never reads images.")
    args = parser.parse_args()

    specs = [
        ("Three-MDOT", "train", args.train_list, args.dataset_root),
        ("Three-MDOT", "val", args.val_list, args.dataset_root),
        ("Three-MDOT", "test", args.test_list, args.dataset_root),
    ] + args.split_spec
    rows: List[dict] = []
    dataset_splits: Dict[str, Dict[str, List[str]]] = collections.defaultdict(dict)
    for dataset, split, list_path, root in specs:
        sequence_names = read_sequence_list(list_path)
        if split in dataset_splits[dataset]:
            raise ValueError("Duplicate split specification: {}:{}".format(dataset, split))
        dataset_splits[dataset][split] = sequence_names
        rows.extend(
            build_sequence_rows(dataset, split, sequence_names, root, args.scan_annotations)
        )

    overlap_rows: List[dict] = []
    for dataset, split_sequences in sorted(dataset_splits.items()):
        dataset_overlap = split_overlap_rows(split_sequences)
        for row in dataset_overlap:
            row["dataset"] = dataset
        overlap_rows.extend(dataset_overlap)
        assert_targets_not_split(split_sequences)

    inventory_rows = dataset_inventory(rows)
    target_rows = target_inventory(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "dataset_inventory.csv",
        inventory_rows,
        ["dataset", "split", "target_count", "sequence_view_count", "frame_count", "all_have_gt", "all_have_occlusion", "all_have_out_of_view"],
    )
    write_csv(
        args.output_dir / "split_target_inventory.csv",
        target_rows,
        ["dataset", "split", "target_id", "view_count", "view_ids", "all_views_grouped", "frame_count", "used_in_training", "reserved_for_test", "eligible_for_dev", "eligibility_reason"],
    )
    write_csv(
        args.output_dir / "split_overlap_audit.csv",
        overlap_rows,
        ["dataset", "left_split", "right_split", "overlap_type", "overlap_count", "overlap_values", "status"],
    )
    write_csv(args.output_dir / "sequence_annotation_inventory.csv", rows, REQUIRED_SEQUENCE_FIELDS)

    config_paths = args.training_config or [
        REPO_ROOT / "experiments" / "entertrack" / "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40.yaml",
        REPO_ROOT / "experiments" / "entertrack" / "pcum_v2_a_weighted_softmax_t010_ep5.yaml",
        REPO_ROOT / "experiments" / "entertrack" / "pcum_v2_d1_visible_safe_rank_softmax_t010_ep5.yaml",
        REPO_ROOT / "experiments" / "entertrack" / "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5.yaml",
    ]
    config_rows = [extract_training_config(path) for path in config_paths]
    (args.output_dir / "current_training_data_audit.md").write_text(
        render_training_report(config_rows), encoding="utf-8"
    )
    (args.output_dir / "dataset_audit_summary.md").write_text(
        render_summary(inventory_rows, overlap_rows, args.scan_annotations), encoding="utf-8"
    )
    print("Wrote dataset audit to {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
