#!/usr/bin/env python3
"""Export one B0-ABC-Plain 2x2 run as compact, auditable artifacts."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from export_b0_abc_plain_results import export_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def network_state_sha256(state):
    """Hash tensor names and bytes, excluding optimizer/config metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def count_lines(path):
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


def export_multiview_summaries(training_log, output_dir):
    records = []
    for line in Path(training_log).read_text(encoding="utf-8").splitlines():
        if line.startswith("[MultiviewEpoch] "):
            records.append(json.loads(line.split(" ", 1)[1]))
    destination = output_dir / "multiview_epoch_summaries.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    frame_rows = []
    for record in records:
        for view, stats in sorted(record.get("frame_sampling", {}).items()):
            frame_rows.append({
                "epoch": record["epoch"],
                "loader": record["loader"],
                "view": view,
                "view_count": record["view_counts"][view],
                "group_samples": record["group_samples"],
                "distinct_view_frame_ratio": record[
                    "distinct_view_frame_ratio"],
                "delta_t_mean": stats["delta_t_mean"],
                "delta_t_median": stats["delta_t_median"],
                "delta_t_p90": stats["delta_t_p90"],
                "unique_pair_count": stats["unique_pair_count"],
                "unique_pair_ratio": stats["unique_pair_ratio"],
                "causal_violations": record["causal_violations"],
                "visibility_violations": record["visibility_violations"],
            })
    if frame_rows:
        with (output_dir / "frame_sampling_epoch_metrics.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(frame_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(frame_rows)

        train_record = next(
            record for record in records if record["loader"] == "train")
        coverage_rows = []
        for target_id, coverage in sorted(
                train_record["visibility_coverage"].items()):
            row = {
                "target_id": target_id,
                "abc_common_visible_frames": coverage[
                    "abc_common_visible_frames"],
                "abc_common_total_frames": coverage[
                    "abc_common_total_frames"],
            }
            for view in ("A", "B", "C"):
                row["{}_visible_frames".format(view)] = coverage["views"][view][
                    "visible_frames"]
                row["{}_total_frames".format(view)] = coverage["views"][view][
                    "total_frames"]
            coverage_rows.append(row)
        with (output_dir / "train_visibility_coverage.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(coverage_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(coverage_rows)
    return records


def export_sampling_manifests(manifest_dir, output_dir, sample_rows):
    inventory = []
    sample = []
    if manifest_dir is None:
        return inventory
    for path in sorted(Path(manifest_dir).glob("*_sampling_epoch_*.jsonl")):
        rows = count_lines(path)
        inventory.append({
            "filename": path.name,
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        if not sample and path.name.startswith("train_sampling_epoch_"):
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= sample_rows:
                        break
                    sample.append(json.loads(line))
    if inventory:
        with (output_dir / "sampling_manifest_inventory.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("filename", "rows", "bytes", "sha256"),
                lineterminator="\n")
            writer.writeheader()
            writer.writerows(inventory)
    if sample:
        with (output_dir / "train_sampling_epoch_0001_sample.jsonl").open(
                "w", encoding="utf-8") as handle:
            for row in sample:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return inventory


def export_result_manifest(results_dir, output_dir):
    rows = []
    excluded_suffixes = ("_time.txt", "_max_score.txt", "_APCE.txt")
    for path in sorted(Path(results_dir).glob("*.txt")):
        if path.name.endswith(excluded_suffixes):
            continue
        rows.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with (output_dir / "tracking_bbox_manifest.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("filename", "bytes", "sha256"),
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--command-file", required=True, type=Path)
    parser.add_argument("--eval-log", required=True, type=Path)
    parser.add_argument("--eval-summary", required=True, type=Path)
    parser.add_argument("--sequence-metrics", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--dataset", default="threemdot_val")
    parser.add_argument("--runid", required=True, type=int)
    parser.add_argument("--sample-manifest-rows", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, metrics = export_metrics(args.checkpoint, args.output_dir)
    multiview = export_multiview_summaries(args.training_log, args.output_dir)
    sampling_inventory = export_sampling_manifests(
        args.manifest_dir, args.output_dir, args.sample_manifest_rows)
    bbox_rows = export_result_manifest(args.results_dir, args.output_dir)

    shutil.copyfile(args.training_log, args.output_dir / "training.log")
    shutil.copyfile(args.command_file, args.output_dir / "launch_command.txt")
    shutil.copyfile(args.eval_log, args.output_dir / "inner_val_run.log")
    shutil.copyfile(args.eval_summary, args.output_dir / "inner_val_summary.json")
    shutil.copyfile(
        args.sequence_metrics, args.output_dir / "inner_val_sequence_metrics.csv")

    summary = json.loads(args.eval_summary.read_text(encoding="utf-8"))
    state = checkpoint.get("net", checkpoint)
    protocol = summary["evaluation_protocol"]
    provenance = {
        "experiment": args.experiment,
        "sampler": args.sampler,
        "config": args.config,
        "dataset": args.dataset,
        "runid": args.runid,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_not_committed": True,
        "training_log_path": str(args.training_log.resolve()),
        "training_log_sha256": sha256(args.training_log),
        "training_epochs": len(metrics["epoch"]),
        "multiview_summary_records": len(multiview),
        "sampling_manifest_files": len(sampling_inventory),
        "sampling_manifest_rows": sum(row["rows"] for row in sampling_inventory),
        "tracking_bbox_files": len(bbox_rows),
        "evaluation_protocol": protocol,
        "evaluation_sequence_count": summary["sequence_count"],
        "no_gt_inference": True,
        "remote_state_source": "none",
        "network_tensor_count": len(state),
        "network_state_sha256": network_state_sha256(state),
        "atp_key_count": sum(".atp." in key.lower() for key in state),
        "pcum_key_count": sum("pcum" in key.lower() for key in state),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
