import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "entertrack"
BASE_CONFIG = CONFIG_DIR / (
    "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_"
    "ep0015_t2_no_gt_normal.yaml"
)


def config_name(mode, temperature=None, ablation="normal"):
    if mode == "mean":
        name = "pcum_v2_a0_mean_ep0015"
    else:
        prefix = "softmax" if mode == "confidence_softmax" else "sigmoid"
        temperature_label = int(round(float(temperature) * 100))
        name = "pcum_v2_a0_{}_t{:03d}_ep0015".format(
            prefix, temperature_label
        )
    if ablation == "zero":
        name += "_zero"
    elif ablation == "temporal_shuffle":
        name += "_delay"
    return name


def write_config(base, mode, temperature, ablation):
    data = copy.deepcopy(base)
    pcum = data["MODEL"]["PCUM"]
    pcum["REMOTE_AGGREGATION"] = mode
    pcum["REMOTE_WEIGHT_TEMPERATURE"] = float(temperature)
    pcum["REMOTE_WEIGHT_EPS"] = 1e-6
    pcum["REMOTE_WEIGHT_MIN_QUALITY"] = 0.0
    pcum["REMOTE_WEIGHT_DIAGNOSTICS"] = True
    test_pcum = data["TEST"]["PCUM"]
    test_pcum["REMOTE_STATE_SOURCE"] = "tracker"
    test_pcum["USE_REMOTE_VISIBLE_MASK"] = False
    test_pcum["KEEP_LOCAL_IF_REMOTE_WORSE"] = False
    test_pcum["REMOTE_ABLATION"] = ablation
    test_pcum["REMOTE_ABLATION_OFFSET"] = 10

    name = config_name(mode, temperature, ablation)
    path = CONFIG_DIR / "{}.yaml".format(name)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    print(path.relative_to(ROOT))


def main():
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    write_config(base, "mean", 0.25, "normal")
    candidates = [
        ("confidence_softmax", 0.1),
        ("confidence_softmax", 0.25),
        ("confidence_softmax", 0.5),
        ("confidence_softmax", 1.0),
        ("confidence_sigmoid", 0.25),
    ]
    for mode, temperature in candidates:
        for ablation in ("normal", "zero", "temporal_shuffle"):
            write_config(base, mode, temperature, ablation)


if __name__ == "__main__":
    main()
