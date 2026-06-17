import argparse

import _init_paths  # noqa: F401
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


LOCAL_CONFIGS = [
    ("pcum_ablation_current_baseline", "baseline"),
    ("pcum_ablation_current_local_view0", "local_view0"),
    ("pcum_ablation_current_local_allviews", "local_allviews"),
    ("pcum_ablation_current_real_target", "real_target_localtest"),
    ("pcum_ablation_current_allviews_equal", "allviews_equal_localtest"),
    ("pcum_ablation_current_a_weight", "a_weight_localtest"),
    ("pcum_ablation_current_dropout", "dropout_localtest"),
    ("pcum_ablation_current_full", "full_localtest"),
    ("pcum_ablation_current_full_crosslayer", "full_crosslayer_localtest"),
]

REMOTE_CONFIGS = [
    ("pcum_ablation_current_real_target_remote", "real_target_remote"),
    ("pcum_ablation_current_allviews_equal_remote", "allviews_equal_remote"),
    ("pcum_ablation_current_a_weight_remote", "a_weight_remote"),
    ("pcum_ablation_current_dropout_remote", "dropout_remote"),
    ("pcum_ablation_current_full_remote", "full_remote"),
    ("pcum_ablation_current_full_crosslayer_remote", "full_crosslayer_remote"),
]


def main():
    parser = argparse.ArgumentParser(description="Analyze current PCUM ablations.")
    parser.add_argument("--dataset_name", default="threemdot_test")
    parser.add_argument("--runid_local", type=int, default=200)
    parser.add_argument("--runid_remote", type=int, default=201)
    args = parser.parse_args()

    trackers = []
    for config_name, display_name in LOCAL_CONFIGS:
        trackers.extend(trackerlist(
            name="entertrack",
            parameter_name=config_name,
            dataset_name=args.dataset_name,
            run_ids=args.runid_local,
            display_name=display_name,
        ))

    for config_name, display_name in REMOTE_CONFIGS:
        trackers.extend(trackerlist(
            name="entertrack",
            parameter_name=config_name,
            dataset_name=args.dataset_name,
            run_ids=args.runid_remote,
            display_name=display_name,
        ))

    dataset = get_dataset(args.dataset_name)
    print_results(
        trackers,
        dataset,
        args.dataset_name,
        merge_results=False,
        plot_types=("success", "norm_prec", "prec"),
    )


if __name__ == "__main__":
    main()
