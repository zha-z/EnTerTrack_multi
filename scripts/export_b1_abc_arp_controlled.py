#!/usr/bin/env python3
"""Export the controlled B1 ARP experiment as compact audit artifacts."""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from export_b0_abc_plain_results import export_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results/b1_abc_arp_controlled_20260819"
B0_RESULTS = ROOT / "docs/results/b0_abc_plain_2x2_20260818"
FORMAL = ROOT / "output/diagnostics/b1_abc_arp/formal_20260819_seed42_4gpu_r001"
SMOKE = ROOT / "output/diagnostics/b1_abc_arp/smoke_20260819_r001"
B1_CKPT_DIR = FORMAL / "checkpoints/train/entertrack/b1_abc_arp_4gpu"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_dir(runid):
    matches = sorted((ROOT / "output/test/tracking_results/entertrack").glob(
        "*_{:d}".format(runid)))
    if len(matches) != 1:
        raise RuntimeError("runid {} result directory count {}".format(
            runid, len(matches)))
    return matches[0]


def prediction_rows(experiment, epoch, runid):
    rows = []
    excluded = ("_time.txt", "_max_score.txt", "_APCE.txt")
    directory = result_dir(runid)
    for path in sorted(directory.glob("*.txt")):
        if path.name.endswith(excluded):
            continue
        rows.append({
            "experiment": experiment,
            "epoch": epoch,
            "runid": runid,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "path": str(path.resolve()),
        })
    if len(rows) != 15:
        raise RuntimeError("{} ep{} predictions: {}".format(
            experiment, epoch, len(rows)))
    return rows


def checkpoint_path(experiment, epoch):
    if experiment == "E1":
        directory = (ROOT / "output/diagnostics/b0_abc_plain_long/"
                     "e1_run_20260818_seed42_4gpu_r002/checkpoints/train/"
                     "entertrack/b0_abc_plain_4gpu_ep50")
    elif experiment == "E3":
        directory = (ROOT / "output/diagnostics/b0_abc_plain_ind_sampler_long/"
                     "e3_run_20260818_seed42_4gpu_r001/checkpoints/train/"
                     "entertrack/b0_abc_plain_ind_sampler_4gpu_ep50")
    elif experiment == "B1":
        directory = B1_CKPT_DIR
    else:
        raise ValueError(experiment)
    return directory / "EnTeRTrack_ep{:04d}.pth.tar".format(epoch)


def pearson(xs, ys):
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_manifest = []
    checkpoint_manifest = []

    with (B0_RESULTS / "inner_val_sweep.csv").open(newline="") as handle:
        prior = {
            (row["experiment"], int(row["epoch"])): row
            for row in csv.DictReader(handle)
        }
    training = {}
    for experiment in ("E1", "E3"):
        with (B0_RESULTS / experiment / "training_epoch_metrics.csv").open(
                newline="") as handle:
            training[experiment] = {
                int(row["epoch"]): row for row in csv.DictReader(handle)
            }

    stage_a_rows = []
    runids = {
        "E1": {25: 28125, 26: 29126, 27: 29127, 28: 29128, 29: 29129, 30: 28130},
        "E3": {25: 28225, 26: 29326, 27: 29327, 28: 29328, 29: 29329, 30: 28330},
    }
    for experiment in ("E1", "E3"):
        for epoch in range(25, 31):
            runid = runids[experiment][epoch]
            if epoch in (26, 27, 28, 29):
                summary_path = OUT / "stage_a" / experiment / (
                    "ep{:04d}/summary.json".format(epoch))
                result_kind = "new"
            else:
                summary_path = B0_RESULTS / experiment / (
                    "inner_val_ep{:04d}/summary.json".format(epoch))
                result_kind = "existing"
            summary = load_summary(summary_path)
            metrics = summary["overall"]
            checkpoint = checkpoint_path(experiment, epoch)
            train = training[experiment][epoch]
            stage_a_rows.append({
                "experiment": experiment,
                "epoch": epoch,
                "auc": metrics["auc"] * 100.0,
                "precision": metrics["precision"] * 100.0,
                "norm_precision": metrics["normalized_precision"] * 100.0,
                "train_loss": train["train_total_loss"],
                "val_loss": train["val_total_loss"],
                "train_iou": train["train_iou"],
                "val_iou": train["val_iou"],
                "head_lr": train["lr_group0"],
                "backbone_lr": train["lr_group1"],
                "per_view_auc": json.dumps({
                    key: value["auc"] * 100.0
                    for key, value in summary["per_view"].items()
                }, sort_keys=True),
                "per_target_auc": json.dumps({
                    key: value["auc"] * 100.0
                    for key, value in summary["per_target"].items()
                }, sort_keys=True),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "runid": runid,
                "prediction_manifest": "prediction_manifest.csv",
                "result_kind": result_kind,
            })
            prediction_manifest.extend(prediction_rows(
                experiment, epoch, runid))
            checkpoint_manifest.append({
                "experiment": experiment,
                "epoch": epoch,
                "path": str(checkpoint.resolve()),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "runid": runid,
            })
    write_csv(OUT / "stage_a_checkpoint_sweep.csv", stage_a_rows)

    final_checkpoint = checkpoint_path("B1", 25)
    export_metrics(final_checkpoint, OUT)
    (OUT / "training_epoch_metrics.csv").replace(
        OUT / "b1_training_epoch_metrics.csv")

    b1_sweep = []
    b1_summaries = {}
    for epoch in (15, 20, 25):
        runid = 29400 + epoch
        summary_path = OUT / "B1" / "inner_val_ep{:04d}".format(epoch) / "summary.json"
        summary = load_summary(summary_path)
        b1_summaries[epoch] = summary
        metrics = summary["overall"]
        checkpoint = checkpoint_path("B1", epoch)
        b1_sweep.append({
            "epoch": epoch,
            "auc": metrics["auc"] * 100.0,
            "precision": metrics["precision"] * 100.0,
            "norm_precision": metrics["normalized_precision"] * 100.0,
            "runid": runid,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
        })
        prediction_manifest.extend(prediction_rows("B1", epoch, runid))
        checkpoint_manifest.append({
            "experiment": "B1",
            "epoch": epoch,
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "runid": runid,
        })
    write_csv(OUT / "b1_inner_val_sweep.csv", b1_sweep)
    write_csv(OUT / "prediction_manifest.csv", prediction_manifest)
    write_csv(OUT / "checkpoint_manifest.csv", checkpoint_manifest)

    e0 = load_summary(B0_RESULTS / "E1/inner_val_ep0025/summary.json")
    b1 = b1_summaries[25]
    view_rows = []
    for view in ("A", "B", "C"):
        e0_auc = e0["per_view"][view]["auc"] * 100.0
        b1_auc = b1["per_view"][view]["auc"] * 100.0
        view_rows.append({
            "view": view, "e0_auc": e0_auc, "b1_auc": b1_auc,
            "delta_auc": b1_auc - e0_auc,
        })
    write_csv(OUT / "b1_per_view_metrics.csv", view_rows)

    arp_records = [
        json.loads(line) for line in
        (FORMAL / "logs/arp_epoch_metrics.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    arp_epoch_rows = []
    sampling_rows = []
    for record in arp_records:
        overall = record["overall"]
        row = {
            "epoch": record["epoch"],
            "loader": record["loader"],
            **overall,
        }
        for view in ("A", "B", "C"):
            row["view_{}_keep_ratio_mean".format(view)] = record[
                "by_view"][view]["keep_ratio_mean"]
            row["view_{}_threshold_mean".format(view)] = record[
                "by_view"][view]["atp_threshold_mean"]
        arp_epoch_rows.append(row)
        sampling_rows.append({
            "epoch": record["epoch"],
            "loader": record["loader"],
            "sampler": "common-visible",
            "independent_view_sampling": False,
            "A_count": record["by_view"]["A"]["samples"],
            "B_count": record["by_view"]["B"]["samples"],
            "C_count": record["by_view"]["C"]["samples"],
            "total_local_samples": overall["samples"],
        })
    write_csv(OUT / "arp_epoch_metrics.csv", arp_epoch_rows)
    write_csv(OUT / "sampling_manifest.csv", sampling_rows)

    val25 = next(record for record in arp_records
                 if record["loader"] == "val" and record["epoch"] == 25)
    target_rows = []
    for target in ("md3016", "md3027", "md3034", "md3048", "md3055"):
        e0_auc = e0["per_target"][target]["auc"] * 100.0
        b1_auc = b1["per_target"][target]["auc"] * 100.0
        delta = b1_auc - e0_auc
        arp = val25["by_target"][target]
        target_rows.append({
            "target": target,
            "e0_auc": e0_auc,
            "b1_auc": b1_auc,
            "delta_auc": delta,
            "effect": "helpful" if delta > 0 else "harmful" if delta < 0 else "tie",
            "bbox_area_mean": arp["bbox_area_mean"],
            "keep_ratio_mean": arp["keep_ratio_mean"],
            "pruned_search_tokens_mean": arp["pruned_search_tokens_mean"],
            "bbox_area_keep_ratio_pearson_samples": arp[
                "bbox_area_keep_ratio_pearson"],
        })
    write_csv(OUT / "b1_per_target_metrics.csv", target_rows)

    sequence_sources = {}
    for label, path in (
            ("e0", B0_RESULTS / "E1/inner_val_ep0025/sequence_metrics.csv"),
            ("b1", OUT / "B1/inner_val_ep0025/sequence_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            sequence_sources[label] = {
                row["sequence"]: row for row in csv.DictReader(handle)
            }
    sequence_rows = []
    for sequence in sorted(sequence_sources["e0"]):
        e0_sequence = sequence_sources["e0"][sequence]
        b1_sequence = sequence_sources["b1"][sequence]
        e0_auc = float(e0_sequence["auc"]) * 100.0
        b1_auc = float(b1_sequence["auc"]) * 100.0
        sequence_rows.append({
            "sequence": sequence,
            "target": e0_sequence["target"],
            "view": e0_sequence["view"],
            "e0_auc": e0_auc,
            "b1_auc": b1_auc,
            "delta_auc": b1_auc - e0_auc,
            "frame_count": e0_sequence["frame_count"],
        })
    write_csv(OUT / "b1_per_sequence_metrics.csv", sequence_rows)

    target_correlations = {
        "n_targets": len(target_rows),
        "bbox_area_vs_keep_ratio_pearson": pearson(
            [row["bbox_area_mean"] for row in target_rows],
            [row["keep_ratio_mean"] for row in target_rows]),
        "bbox_area_vs_auc_delta_pearson": pearson(
            [row["bbox_area_mean"] for row in target_rows],
            [row["delta_auc"] for row in target_rows]),
        "keep_ratio_vs_auc_delta_pearson": pearson(
            [row["keep_ratio_mean"] for row in target_rows],
            [row["delta_auc"] for row in target_rows]),
    }
    (OUT / "target_size_keep_ratio_correlation.json").write_text(
        json.dumps(target_correlations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    checkpoint = torch.load(str(final_checkpoint), map_location="cpu")
    state = checkpoint["net"]
    pretrain = ROOT / "pretrained_models/tiny_7_OSTrack_ep0300.pth.tar"
    identity = {
        "experiment": "B1-ABC-ARP",
        "backbone": "vit_tiny_patch16_224_arp",
        "embedding_dim": 192,
        "depth": 6,
        "heads": 3,
        "template_tokens": 64,
        "initial_search_tokens": 256,
        "restored_search_tokens": 256,
        "ce_loc": [0],
        "ce_keep_ratio": [0.7],
        "pruning_enabled": True,
        "dynamic_threshold_enabled": True,
        "token_compensation_enabled": True,
        "flat_multiview": True,
        "common_visible_sampler": True,
        "independent_view_sampling": False,
        "pcum": False,
        "c3r": False,
        "fcvc": False,
        "remote_state": False,
        "pretrain_path": str(pretrain.resolve()),
        "pretrain_sha256": sha256(pretrain),
        "checkpoint_path": str(final_checkpoint.resolve()),
        "checkpoint_sha256": sha256(final_checkpoint),
        "network_tensor_count": len(state),
        "atp_key_count": sum(".atp." in key for key in state),
        "pcum_key_count": sum("pcum" in key.lower() for key in state),
    }
    (OUT / "network_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    protocol = b1["evaluation_protocol"]
    provenance = {
        "branch": "feature/pcum-cross-layer-arp",
        "training_run": str(FORMAL.resolve()),
        "smoke_run": str(SMOKE.resolve()),
        "seed": 42,
        "gpus": [0, 1, 2, 3],
        "batch_size_per_gpu": 2,
        "epochs": 25,
        "samples_per_epoch": 6000,
        "dataset": "threemdot_val",
        "official_test_run": False,
        "no_gt_inference": True,
        "remote_state_source": "none",
        "evaluation_protocol": protocol,
        "stage_a_new_runids": [29126, 29127, 29128, 29129,
                                   29326, 29327, 29328, 29329],
        "b1_runids": [29415, 29420, 29425],
        "formal_checkpoint_count": 25,
        "formal_training_log_sha256": sha256(
            FORMAL / "logs/entertrack-b1_abc_arp_4gpu.log"),
        "arp_epoch_metrics_sha256": sha256(
            FORMAL / "logs/arp_epoch_metrics.jsonl"),
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    shutil.copyfile(
        FORMAL / "logs/entertrack-b1_abc_arp_4gpu.log",
        OUT / "B1/training.log")
    shutil.copyfile(
        FORMAL / "logs/arp_epoch_metrics.jsonl",
        OUT / "B1/arp_epoch_metrics.jsonl")
    shutil.copyfile(
        SMOKE / "logs/entertrack-b1_abc_arp_4gpu_smoke.log",
        OUT / "smoke.log")

    print(json.dumps({
        "stage_a_rows": len(stage_a_rows),
        "b1_sweep_rows": len(b1_sweep),
        "prediction_rows": len(prediction_manifest),
        "checkpoint_rows": len(checkpoint_manifest),
        "target_correlations": target_correlations,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
