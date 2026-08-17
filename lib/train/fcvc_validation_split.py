"""Frozen target-level split for FCVC train-side validation."""

import hashlib
import json
import random
from pathlib import Path

from lib.train.data.fcvc_sampler import read_synchronized_frame_pool


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _frame_count(dataset_root, target, view):
    path = Path(dataset_root) / target / "{}-{}".format(target, view) / "groundtruth.txt"
    if not path.is_file():
        raise FileNotFoundError("missing official-train annotation: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _split_stats(targets, frame_pool, dataset_root):
    per_target = {}
    total_frames = 0
    for target in targets:
        view_frames = {
            view: _frame_count(dataset_root, target, view)
            for view in (1, 2, 3)
        }
        total_frames += sum(view_frames.values())
        per_target[target] = {
            "view_frames": view_frames,
            "synchronized_frames": len(frame_pool[target]),
            "total_view_frames": sum(view_frames.values()),
        }
    return {
        "target_count": len(targets),
        "view_count": len(targets) * 3,
        "synchronized_frame_count": sum(
            len(frame_pool[target]) for target in targets),
        "total_frames": total_frames,
        "per_target": per_target,
    }


def build_target_split(source_manifest, dataset_root, seed=42,
                       train_count=18, val_count=4):
    if int(seed) != 42:
        raise ValueError("FCVC validation split seed must remain 42")
    frame_pool = read_synchronized_frame_pool(source_manifest)
    targets = sorted(frame_pool)
    if len(targets) != int(train_count) + int(val_count):
        raise ValueError("expected exactly 22 legal official-train targets")
    validation_targets = sorted(random.Random(seed).sample(targets, val_count))
    validation_set = set(validation_targets)
    train_targets = [target for target in targets if target not in validation_set]
    if set(train_targets).intersection(validation_set):
        raise RuntimeError("train/validation target leakage")
    payload = {
        "schema": "fcvc_target_split_v1",
        "source": "threemdot_official_train_legal_sync_targets",
        "algorithm": "sorted_names_then_python_random_sample",
        "split_seed": int(seed),
        "bind_abc_views": True,
        "train_targets": train_targets,
        "validation_targets": validation_targets,
        "intersection": [],
        "train": _split_stats(train_targets, frame_pool, dataset_root),
        "validation": _split_stats(
            validation_targets, frame_pool, dataset_root),
    }
    encoded = json.dumps(
        payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return payload, sha256_bytes(encoded), encoded


def ensure_target_split(output_dir, source_manifest, dataset_root, seed=42,
                        train_count=18, val_count=4):
    output_dir = Path(output_dir)
    payload, digest, encoded = build_target_split(
        source_manifest, dataset_root, seed, train_count, val_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train_targets.txt": (
            "\n".join(payload["train_targets"]) + "\n").encode("utf-8"),
        "val_targets.txt": (
            "\n".join(payload["validation_targets"]) + "\n").encode("utf-8"),
        "target_split.json": encoded,
        "target_split_sha256.txt": (digest + "\n").encode("utf-8"),
    }
    for name, content in files.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError("refusing to overwrite a different target split: {}".format(path))
        if not path.exists():
            path.write_bytes(content)
    return payload, digest
