#!/usr/bin/env python3
"""Export compact, auditable B0-ABC-Plain training and test artifacts."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch


TRAINING_FIELDS = {
    "total_loss": "Loss/total",
    "giou_loss": "Loss/giou",
    "l1_loss": "Loss/l1",
    "focal_loss": "Loss/location",
    "iou": "IoU",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def epoch_history(meter, epochs):
    history = list(meter.history)
    if len(history) == epochs:
        return history
    if len(history) % epochs != 0:
        raise RuntimeError(
            "cannot map history of length {} to {} epochs".format(
                len(history), epochs
            )
        )
    stride = len(history) // epochs
    return [history[(index + 1) * stride - 1] for index in range(epochs)]


def export_metrics(checkpoint, output_dir):
    payload = torch.load(str(checkpoint), map_location="cpu")
    epochs = int(payload["epoch"])
    stats = payload["stats"]
    columns = {"epoch": list(range(1, epochs + 1))}
    for loader in ("train", "val"):
        for output_name, stat_name in TRAINING_FIELDS.items():
            columns["{}_{}".format(loader, output_name)] = epoch_history(
                stats[loader][stat_name], epochs
            )
    for group_index in range(2):
        stat_name = "LearningRate/group{}".format(group_index)
        if stat_name in stats["train"]:
            columns["lr_group{}".format(group_index)] = epoch_history(
                stats["train"][stat_name], epochs
            )
    fieldnames = list(columns)
    with (output_dir / "training_epoch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for index in range(epochs):
            writer.writerow({name: columns[name][index] for name in fieldnames})
    return payload, columns


def export_multiview_counts(training_log, output_dir):
    records = []
    for line in Path(training_log).read_text(encoding="utf-8").splitlines():
        if not line.startswith("[MultiviewEpoch] "):
            continue
        records.append(json.loads(line.split(" ", 1)[1]))
    with (output_dir / "multiview_epoch_counts.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def export_result_manifest(results_dir, output_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("*.txt")):
        rows.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with (output_dir / "tracking_results_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--test-log", required=True, type=Path)
    parser.add_argument("--test-summary", required=True, type=Path)
    parser.add_argument("--sequence-metrics", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runid", required=True, type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, metrics = export_metrics(args.checkpoint, args.output_dir)
    multiview = export_multiview_counts(args.training_log, args.output_dir)
    result_rows = export_result_manifest(args.results_dir, args.output_dir)
    shutil.copyfile(args.test_log, args.output_dir / "test_run.log")
    shutil.copyfile(args.test_summary, args.output_dir / "test_summary.json")
    (args.output_dir / "sequence_metrics.csv").write_text(
        args.sequence_metrics.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    net = checkpoint["net"]
    summary = json.loads(args.test_summary.read_text(encoding="utf-8"))
    evaluation = summary.get("evaluation_protocol", {})
    manifest = {
        "config": args.config,
        "dataset": args.dataset,
        "evaluation_protocol": evaluation.get("name"),
        "evaluator": evaluation.get("implementation"),
        "evaluation_success_comparator": evaluation.get("success_comparator"),
        "evaluation_uses_target_visible": evaluation.get("uses_target_visible"),
        "evaluation_exclude_invalid_frames": evaluation.get("exclude_invalid_frames"),
        "runid": args.runid,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_not_committed": True,
        "network_tensor_count": len(net),
        "atp_key_count": sum(".atp." in key.lower() for key in net),
        "pcum_key_count": sum("pcum" in key.lower() for key in net),
        "training_log_sha256": sha256(args.training_log),
        "test_log_sha256": sha256(args.test_log),
        "training_epochs": len(metrics["epoch"]),
        "multiview_record_count": len(multiview),
        "test_sequence_count": summary["sequence_count"],
        "tracking_result_file_count": len(result_rows),
        "no_gt_inference": True,
        "remote_state_source": "none",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
