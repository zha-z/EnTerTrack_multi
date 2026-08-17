#!/usr/bin/env python3
"""Generate fixed D2-G0 epoch-sweep validation configs."""

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "entertrack" / (
    "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5.yaml")
PREFIX = "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010"


def main():
    with SOURCE.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    for epoch in range(1, 6):
        for mode, use_remote in (("t1", False), ("t2_raw", True)):
            config = copy.deepcopy(base)
            config["MODEL"]["PRETRAIN_FILE"] = "tiny_7_OSTrack_ep0300.pth.tar"
            config["TEST"]["SAVE_DIR"] = (
                "output/pcum_v2_d2_g0_remote_suppression_ep5")
            config["TEST"]["CHECKPOINT_NAME"] = (
                "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5")
            config["TEST"]["EPOCH"] = epoch
            test_pcum = config["TEST"]["PCUM"]
            test_pcum["USE_REMOTE"] = use_remote
            test_pcum["REMOTE_STATE_SOURCE"] = "tracker"
            test_pcum["USE_REMOTE_VISIBLE_MASK"] = False
            test_pcum["REMOTE_ABLATION"] = "normal"
            name = "{}_ep{}_{}_val.yaml".format(PREFIX, epoch, mode)
            path = SOURCE.parent / name
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
