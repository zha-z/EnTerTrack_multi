"""One-time fixed synchronized pair-validation manifest and partitioning."""

import csv
import hashlib
import io
import random
from collections import Counter, defaultdict
from pathlib import Path

from lib.train.data.fcvc_sampler import RECEIVER_LAYOUT, read_synchronized_frame_pool


FIELDS = (
    "case_index", "sync_group_index", "sync_group_id", "target",
    "template_frame", "search_frame", "frame", "receiver", "sender_1",
    "sender_2", "split", "uses_gt_in_student_input",
    "synchronization_validity", "manifest_seed", "random_augmentation",
    "center_jitter", "scale_jitter",
)


def _csv_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _sample_pair(frames, rng, max_interval):
    template = rng.choice(frames)
    candidates = [frame for frame in frames if abs(frame - template) <= max_interval]
    alternatives = [frame for frame in candidates if frame != template]
    return template, rng.choice(alternatives or candidates)


def generate_validation_rows(source_manifest, validation_targets, groups=504,
                             seed=4242, max_interval=200):
    targets = tuple(sorted(validation_targets))
    if len(targets) != 4:
        raise ValueError("pair validation requires exactly four targets")
    if int(groups) != 504 or int(seed) != 4242:
        raise ValueError("fixed pair validation requires groups=504 and seed=4242")
    pool = read_synchronized_frame_pool(source_manifest)
    if not set(targets).issubset(pool):
        raise ValueError("validation target missing from synchronized frame pool")
    rng = random.Random(seed)
    schedule = list(targets) * (groups // len(targets))
    rng.shuffle(schedule)
    rows = []
    for group_index, target in enumerate(schedule):
        template, search = _sample_pair(pool[target], rng, max_interval)
        for receiver_index, (receiver, sender_1, sender_2) in enumerate(RECEIVER_LAYOUT):
            rows.append({
                "case_index": group_index * 3 + receiver_index,
                "sync_group_index": group_index,
                "sync_group_id": "val-g{:04d}".format(group_index),
                "target": target,
                "template_frame": template,
                "search_frame": search,
                "frame": search,
                "receiver": receiver,
                "sender_1": sender_1,
                "sender_2": sender_2,
                "split": "fixed_target_val",
                "uses_gt_in_student_input": False,
                "synchronization_validity": True,
                "manifest_seed": seed,
                "random_augmentation": False,
                "center_jitter": 0.0,
                "scale_jitter": 0.0,
            })
    return rows


def ensure_validation_manifest(validation_dir, source_manifest,
                               validation_targets):
    validation_dir = Path(validation_dir)
    rows = generate_validation_rows(source_manifest, validation_targets)
    encoded = _csv_bytes(rows)
    digest = hashlib.sha256(encoded).hexdigest()
    validation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = validation_dir / "val_pair_manifest.csv"
    sha_path = validation_dir / "val_pair_manifest_sha256.txt"
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise RuntimeError("fixed validation manifest differs from existing file")
    if sha_path.exists() and sha_path.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError("fixed validation manifest SHA differs from existing file")
    if not manifest_path.exists():
        manifest_path.write_bytes(encoded)
    if not sha_path.exists():
        sha_path.write_text(digest + "\n", encoding="utf-8")
    return manifest_path, digest, rows


class FixedPairValidationSampler:
    def __init__(self, manifest_path, validation_targets, world_size=6):
        self.manifest_path = Path(manifest_path)
        self.validation_targets = set(validation_targets)
        self.world_size = int(world_size)
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.full_rows = list(csv.DictReader(handle))
        if len(self.full_rows) != 1512:
            raise ValueError("pair validation manifest must contain 1512 cases")
        if {row["target"] for row in self.full_rows} != self.validation_targets:
            raise ValueError("pair validation manifest target leakage")
        if any(row["uses_gt_in_student_input"].lower() != "false"
               for row in self.full_rows):
            raise ValueError("pair validation student input contains GT")
        self.sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.rows = []

    def partition(self, rank, world_size=6):
        rank, world_size = int(rank), int(world_size)
        if world_size != self.world_size or world_size != 6:
            raise ValueError("pair validation requires world_size=6")
        groups = 504
        groups_per_rank = groups // world_size
        start, end = rank * groups_per_rank, (rank + 1) * groups_per_rank
        self.rows = self.full_rows[start * 3:end * 3]
        if len(self.rows) != 252:
            raise RuntimeError("pair validation rank partition must contain 252 cases")
        receivers = Counter(row["receiver"] for row in self.rows)
        if receivers != Counter({"A": 84, "B": 84, "C": 84}):
            raise RuntimeError("pair validation receiver partition is unbalanced")
        return self.rows

    def audit(self):
        groups = defaultdict(list)
        for row in self.full_rows:
            groups[row["sync_group_id"]].append(row)
        return {
            "case_count": len(self.full_rows),
            "group_count": len(groups),
            "receiver_counts": dict(Counter(
                row["receiver"] for row in self.full_rows)),
            "target_group_counts": dict(Counter(
                row["target"] for row in self.full_rows[::3])),
            "max_interval": max(abs(
                int(row["search_frame"]) - int(row["template_frame"]))
                for row in self.full_rows),
            "uses_gt_in_student_input_count": sum(
                row["uses_gt_in_student_input"].lower() != "false"
                for row in self.full_rows),
            "sha256": self.sha256,
        }
