#!/usr/bin/env python3
"""Read-only real-data parity audit for the paired LSPCA-PCUM controls."""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config.entertrack.config import get_default_config, update_config_from_file  # noqa: E402
from lib.train.admin.settings import Settings  # noqa: E402
from lib.train.base_functions import build_dataloaders_threemdot, update_settings  # noqa: E402
from lib.train.data.sampler_threemdot import TrackingSamplerThreeMDOT  # noqa: E402
from lib.train.train_script import use_grouped_multiview_loader  # noqa: E402


CONFIG_DIR = ROOT / "experiments/entertrack"
J0_PATTERN = "ostrack_deit_tiny_b0_j0v2_lspca_control_ep15_fold%d"
J1_PATTERN = "ostrack_deit_tiny_b0_j1v2_lspca_pcum_ep15_fold%d"


def load_config(name):
    config = get_default_config()
    update_config_from_file(str(CONFIG_DIR / (name + ".yaml")), base_cfg=config)
    # The audit reads exactly one sample in-process. These audit-only overrides
    # do not mutate the YAML or the declared training protocol.
    config.TRAIN.NUM_WORKER = 0
    config.TRAIN.BATCH_SIZE = 1
    config.DATA.TRAIN.SAMPLE_PER_EPOCH = 1
    config.DATA.VAL.SAMPLE_PER_EPOCH = 1
    return config


def build_train_dataset(config):
    settings = Settings()
    settings.local_rank = -1
    settings.use_lmdb = False
    update_settings(settings, config)
    train_loader, _ = build_dataloaders_threemdot(config, settings)
    if not isinstance(train_loader.dataset, TrackingSamplerThreeMDOT):
        raise RuntimeError("Resolved loader is not TrackingSamplerThreeMDOT")
    return train_loader.dataset


def attach_trace(dataset):
    traces = []
    for source in dataset.datasets:
        original = source.get_frames

        def recorded(seq_id, frame_ids, anno=None, _source=source,
                     _original=original):
            traces.append({
                "sequence": str(_source.sequence_list[seq_id]),
                "frame_ids": [int(value) for value in frame_ids],
            })
            return _original(seq_id, frame_ids, anno)

        source.get_frames = recorded
    return traces


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_digest(value):
    digest = hashlib.sha256()

    def update(item, path="root"):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(path.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                update(item[key], path + "." + str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                update(child, path + "[%d]" % index)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update((path + "=" + repr(item)).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def audit(fold, seed):
    j0_name = J0_PATTERN % fold
    j1_name = J1_PATTERN % fold
    j0_cfg = load_config(j0_name)
    j1_cfg = load_config(j1_name)
    if not use_grouped_multiview_loader(j0_cfg):
        raise RuntimeError("J0v2 does not resolve to grouped multiview loader")
    if not use_grouped_multiview_loader(j1_cfg):
        raise RuntimeError("J1v2 does not resolve to grouped multiview loader")
    if j0_cfg.DATA.TRAIN.SPLIT_FILE != j1_cfg.DATA.TRAIN.SPLIT_FILE:
        raise RuntimeError("Train split files differ")
    if j0_cfg.DATA.VAL.SPLIT_FILE != j1_cfg.DATA.VAL.SPLIT_FILE:
        raise RuntimeError("Holdout split files differ")

    j0_dataset = build_train_dataset(j0_cfg)
    j1_dataset = build_train_dataset(j1_cfg)
    j0_trace = attach_trace(j0_dataset)
    j1_trace = attach_trace(j1_dataset)

    reset_seed(seed)
    j0_sample = j0_dataset[0]
    reset_seed(seed)
    j1_sample = j1_dataset[0]

    j0_digest = tensor_digest(j0_sample)
    j1_digest = tensor_digest(j1_sample)
    report = {
        "status": "PASS" if j0_trace == j1_trace and j0_digest == j1_digest else "FAIL",
        "fold": int(fold),
        "seed": int(seed),
        "sampler_class_j0": type(j0_dataset).__name__,
        "sampler_class_j1": type(j1_dataset).__name__,
        "train_split": str(j0_cfg.DATA.TRAIN.SPLIT_FILE),
        "holdout_split": str(j0_cfg.DATA.VAL.SPLIT_FILE),
        "j0_trace": copy.deepcopy(j0_trace),
        "j1_trace": copy.deepcopy(j1_trace),
        "j0_processed_sample_sha256": j0_digest,
        "j1_processed_sample_sha256": j1_digest,
        "template_shape": list(j0_sample["template_images"].shape),
        "search_shape": list(j0_sample["search_images"].shape),
        "uses_threemdot_val_or_test": False,
    }
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    print(json.dumps(audit(args.fold, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
