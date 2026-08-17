"""Temporal Gate v2 fold1 split and immutable behavior-rollout helpers.

The module contains no dataset loader.  A later authorized rollout driver must
provide already computed prediction-only rows to ``PredictionTableWriter``;
GT labels are joined only after that table has been closed and digested.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from lib.train.dataset.c3r_temporal_gate import (
    assert_prediction_row,
    audit_split,
    build_fold0_inner_split,
    build_fold1_inner_split,
    read_id_file,
)


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "lib/train/data_specs/threemdot"
OUTER_TRAIN = SPECS / "c3r_f0_train.txt"
OUTER_HOLDOUT = SPECS / "c3r_f0_holdout.txt"
INNER_TRAIN = SPECS / "c3r_f0_temporal_inner_train.txt"
INNER_DEV = SPECS / "c3r_f0_temporal_inner_dev.txt"
FOLD1_OUTER_TRAIN = SPECS / "c3r_f1_train.txt"
FOLD1_OUTER_HOLDOUT = SPECS / "c3r_f1_holdout.txt"
FOLD1_INNER_TRAIN = SPECS / "c3r_f1_temporal_v2_inner_train.txt"
FOLD1_INNER_DEV = SPECS / "c3r_f1_temporal_v2_inner_dev.txt"
FOLD1_SPLIT_SHA256 = SPECS / "c3r_f1_temporal_v2_split_sha256.json"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_fold0_inner_split(train_path=OUTER_TRAIN,
                               holdout_path=OUTER_HOLDOUT,
                               train_output=INNER_TRAIN,
                               dev_output=INNER_DEV) -> Dict[str, object]:
    outer_train = read_id_file(str(train_path))
    outer_holdout = read_id_file(str(holdout_path))
    inner_train, inner_dev, ranked = build_fold0_inner_split(outer_train)
    audit = audit_split(inner_train, inner_dev, outer_holdout)
    Path(train_output).write_text("\n".join(inner_train) + "\n", encoding="utf-8")
    Path(dev_output).write_text("\n".join(inner_dev) + "\n", encoding="utf-8")
    audit.update({
        "ranking": ranked,
        "inner_train_sequences": len(inner_train),
        "inner_dev_sequences": len(inner_dev),
        "inner_train_sha256": sha256_file(str(train_output)),
        "inner_dev_sha256": sha256_file(str(dev_output)),
    })
    return audit


def generate_fold1_inner_split(
        train_path=FOLD1_OUTER_TRAIN,
        holdout_path=FOLD1_OUTER_HOLDOUT,
        train_output=FOLD1_INNER_TRAIN,
        dev_output=FOLD1_INNER_DEV,
        manifest_output=FOLD1_SPLIT_SHA256) -> Dict[str, object]:
    """Read IDs only and preregister the v2 fold1 target-group split."""
    outer_train = read_id_file(str(train_path))
    outer_holdout = read_id_file(str(holdout_path))
    inner_train, inner_dev, ranked = build_fold1_inner_split(outer_train)
    audit = audit_split(inner_train, inner_dev, outer_holdout)
    Path(train_output).write_text(
        "\n".join(inner_train) + "\n", encoding="utf-8")
    Path(dev_output).write_text(
        "\n".join(inner_dev) + "\n", encoding="utf-8")
    manifest = dict(audit)
    manifest.update({
        "protocol": "temporal-gate-v2",
        "fold_id": 1,
        "ranking_key": "SHA256(temporal-gate-v2|fold1|<target_id>)",
        "ranking": ranked,
        "inner_dev_target_count": 4,
        "inner_train_sequences": len(inner_train),
        "inner_dev_sequences": len(inner_dev),
        "outer_train_ids_sha256": sha256_file(str(train_path)),
        "outer_holdout_ids_sha256": sha256_file(str(holdout_path)),
        "inner_train_sha256": sha256_file(str(train_output)),
        "inner_dev_sha256": sha256_file(str(dev_output)),
        "id_only_generation": True,
        "images_gt_predictions_metrics_read": False,
    })
    Path(manifest_output).write_text(json.dumps(
        manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_output)
    manifest["manifest_sha256"] = sha256_file(str(manifest_output))
    return manifest


class PredictionTableWriter:
    """Write prediction-only rows, then freeze the exact byte stream."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self._closed = False
        self._last_frame = {}

    def append(self, row: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("prediction table is immutable after close")
        assert_prediction_row(row)
        key = (str(row["target_id"]), int(row["receiver_id"]),
               int(row["sender_id"]))
        frame = int(row["frame_id"])
        if key in self._last_frame and frame <= self._last_frame[key]:
            raise ValueError("prediction rows must be strictly increasing per stream")
        self._last_frame[key] = frame
        self._handle.write(json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"),
            allow_nan=False) + "\n")

    def close(self) -> Dict[str, object]:
        if not self._closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True
        digest = sha256_file(str(self.path))
        manifest = {
            "path": str(self.path),
            "sha256": digest,
            "uses_gt": False,
            "streams": len(self._last_frame),
        }
        manifest_path = self.path.with_suffix(self.path.suffix + ".sha256.json")
        manifest_path.write_text(json.dumps(
            manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return manifest

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self._handle.close()


def write_label_table(prediction_path: str, prediction_digest: str,
                      label_rows: Iterable[Mapping[str, object]],
                      output_path: str) -> Dict[str, object]:
    """Persist a separate post-rollout label table after digest verification."""
    if sha256_file(prediction_path) != str(prediction_digest):
        raise RuntimeError("prediction table digest must be recorded before labels")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in label_rows:
        row = dict(source)
        if bool(row.get("uses_gt_for_features", True)):
            raise ValueError("label join must declare uses_gt_for_features=false")
        if row.get("label_status", "valid") != "valid":
            raise ValueError("v2 requires a valid continuous label on every row")
        if "delta_diou" not in row:
            raise ValueError("v2 label table requires delta_diou")
        delta_diou = float(row["delta_diou"])
        if not -2.0 <= delta_diou <= 2.0 or not math.isfinite(delta_diou):
            raise ValueError("delta_diou must be finite and within [-2,2]")
        if "delta_iou" in row and not math.isfinite(float(row["delta_iou"])):
            raise ValueError("auxiliary delta_iou must be finite when present")
        rows.append(row)
    path.write_text("".join(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows), encoding="utf-8")
    return {
        "path": str(path),
        "sha256": sha256_file(str(path)),
        "prediction_sha256": str(prediction_digest),
        "rows": len(rows),
    }


def consolidate_prediction_rows(results_dir: str, output_path: str,
                                checkpoint_digest: str,
                                config_digest: str) -> Dict[str, object]:
    """Freeze accepted per-sender instrumentation as the model input table."""
    files = sorted(Path(results_dir).glob(
        "*_c3r_source_instrumentation.jsonl.gz"))
    if not files:
        raise RuntimeError("rollout produced no Temporal Gate source rows")
    writer = PredictionTableWriter(output_path)
    try:
        for path in files:
            with gzip.open(str(path), "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    source = json.loads(line)
                    row = dict(source)
                    row.update({
                        "receiver_id": int(row.pop("receiver_view")),
                        "sender_id": int(row.pop("sender_view")),
                        "normalized_features": row[
                            "reliability_input_normalized"],
                        "model_input_fields": ["normalized_features"],
                        "packet_accepted": True,
                        "uses_gt": False,
                        "checkpoint_sha256": checkpoint_digest,
                        "config_sha256": config_digest,
                    })
                    row.pop("uses_gt_for_features", None)
                    writer.append(row)
        return writer.close()
    except Exception:
        writer._handle.close()
        raise


def run_authorized_behavior_rollout(config: str, runid: str,
                                    checkpoint: str, split: str,
                                    output_path: str,
                                    counterfactual_diagnostics: bool = False,
                                    remote_information_diagnostics: bool = False,
                                    fold_id: int = 1) -> Dict[str, object]:
    """Run frozen C1 through the existing three-view runner when authorized."""
    if os.environ.get("TEMPORAL_GATE_REAL_ROLLOUT_AUTHORIZED") != "1":
        raise RuntimeError(
            "set TEMPORAL_GATE_REAL_ROLLOUT_AUTHORIZED=1 only in the separately "
            "authorized one-shot rollout stage")
    from lib.test.evaluation import Tracker
    from lib.test.evaluation.data import Sequence, SequenceList
    from lib.test.evaluation.environment import env_settings
    from lib.test.evaluation.running import run_mdot_dataset_three

    allowed = set(read_id_file(split))
    holdout = set(read_id_file(str(FOLD1_OUTER_HOLDOUT)))
    if allowed & holdout:
        raise RuntimeError("outer holdout leaked into requested rollout split")
    # Construct a deliberately prediction-only sequence list.  It reads only
    # image filenames and the ordinary frame-0 initialization box; it never
    # loads later GT, visibility, occlusion, or out-of-view annotations.
    base_path = Path(env_settings().threemdot_val_path)
    selected = SequenceList()
    for sequence_name in sorted(allowed):
        target = sequence_name.rsplit("-", 1)[0]
        sequence_dir = base_path / target / sequence_name
        frames = sorted(str(path) for path in (sequence_dir / "img").glob("*.jpg"))
        if not frames:
            raise RuntimeError("prediction-only sequence has no frame filenames")
        with (sequence_dir / "groundtruth.txt").open(
                "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        init_box = [float(value) for value in first_line.split(",")]
        if len(init_box) != 4 or not all(
                value == value and abs(value) != float("inf")
                for value in init_box):
            raise RuntimeError("invalid frame-0 initialization box")
        selected.append(Sequence(
            sequence_name, frames, "threemdot_train_prediction_only",
            ground_truth_rect=None,
            init_data={0: {"bbox": init_box}},
            target_visible=None,
        ))
    selected_names = {sequence.name for sequence in selected}
    if selected_names != allowed:
        raise RuntimeError("requested rollout split does not match train dataset IDs")
    tracker = Tracker(
        "entertrack", config, "threemdot_train", runid,
        checkpoint_override=checkpoint,
        no_gt_inference=True,
        c3r_instrumentation=True,
        instrumentation_fold_id=int(fold_id),
        temporal_gate_rollout_capture=True,
        temporal_gate_counterfactual_diagnostics=bool(
            counterfactual_diagnostics),
        remote_information_diagnostics=bool(
            remote_information_diagnostics),
    )
    tracker.reserve_results_dir()
    run_mdot_dataset_three(selected, [tracker], debug=False, threads=0,
                           num_gpus=1)
    config_path = ROOT / "experiments/entertrack" / (config + ".yaml")
    return consolidate_prediction_rows(
        tracker.results_dir, output_path,
        checkpoint_digest=sha256_file(checkpoint),
        config_digest=sha256_file(str(config_path)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-fold1-split-only", action="store_true",
        help="read ID text files only; never load a sequence or checkpoint")
    parser.add_argument("--execute-authorized-rollout", action="store_true")
    parser.add_argument("--config", default="entertrack_c3r_temporal_gate_v2_f1")
    parser.add_argument("--runid", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--counterfactual-diagnostics", action="store_true")
    parser.add_argument(
        "--remote-information-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.build_fold1_split_only == args.execute_authorized_rollout:
        raise SystemExit(
            "choose exactly one of --build-fold1-split-only or "
            "--execute-authorized-rollout")
    if args.build_fold1_split_only:
        report = generate_fold1_inner_split()
    else:
        if not all((args.runid, args.checkpoint, args.split, args.output)):
            raise SystemExit("authorized rollout requires runid/checkpoint/split/output")
        report = run_authorized_behavior_rollout(
            args.config, args.runid, args.checkpoint, args.split, args.output,
            counterfactual_diagnostics=args.counterfactual_diagnostics,
            remote_information_diagnostics=
                args.remote_information_diagnostics,
            fold_id=1)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
