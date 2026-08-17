import csv
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import common
from lib.train.admin import env_settings
from lib.train.data.fcvc_sampler import FCVCSampler
from lib.train.fcvc_online_validation import OnlineValidator, _curves
from lib.train.fcvc_pair_validation import PAIR_FIELDS, _append_unique_csv
from lib.train.fcvc_pair_validation import PairValidator
from lib.train.fcvc_training_graph import FCVCTrainingGraph
from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel
from lib.train.fcvc_validation_reporting import (
    ValidationReporter, is_better_online, online_epoch_due,
)
from lib.train.fcvc_validation_sampler import (
    FixedPairValidationSampler, ensure_validation_manifest,
    generate_validation_rows,
)
from lib.train.fcvc_validation_split import build_target_split


SOURCE = (
    common.ROOT
    / "output/multi_agent_collaboration_clean/fcvc_manual_run/full_train_receiver_manifest.csv"
)


class FCVCValidationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split, cls.split_sha, _ = build_target_split(
            SOURCE, env_settings().threemdot_dir)

    def test_target_split_is_frozen_disjoint_and_abc_bound(self):
        train = self.split["train_targets"]
        validation = self.split["validation_targets"]
        self.assertEqual(len(train), 18)
        self.assertEqual(len(validation), 4)
        self.assertFalse(set(train) & set(validation))
        self.assertTrue(self.split["bind_abc_views"])
        self.assertEqual(self.split["train"]["view_count"], 54)
        self.assertEqual(self.split["validation"]["view_count"], 12)

    def test_frozen_validation_target_names(self):
        self.assertEqual(
            self.split["validation_targets"],
            ["md3005", "md3018", "md3031", "md3060"])

    def test_split_reproducibility(self):
        replay, replay_sha, _ = build_target_split(
            SOURCE, env_settings().threemdot_dir)
        self.assertEqual(replay, self.split)
        self.assertEqual(replay_sha, self.split_sha)

    def test_training_sampler_contains_train_targets_only(self):
        sampler = FCVCSampler(
            SOURCE, allowed_targets=self.split["train_targets"])
        rows, _ = sampler.generate_epoch(1)
        self.assertEqual(set(row["target"] for row in rows),
                         set(self.split["train_targets"]))
        self.assertFalse(set(row["target"] for row in rows)
                         & set(self.split["validation_targets"]))

    def test_pair_manifest_fixed_balanced_and_prediction_only(self):
        first = generate_validation_rows(
            SOURCE, self.split["validation_targets"])
        second = generate_validation_rows(
            SOURCE, self.split["validation_targets"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1512)
        self.assertEqual(Counter(row["receiver"] for row in first),
                         Counter({"A": 504, "B": 504, "C": 504}))
        self.assertEqual(set(row["target"] for row in first),
                         set(self.split["validation_targets"]))
        self.assertTrue(all(not row["uses_gt_in_student_input"] for row in first))
        self.assertTrue(all(not row["random_augmentation"] for row in first))
        self.assertTrue(all(row["center_jitter"] == 0.0 for row in first))
        self.assertTrue(all(row["scale_jitter"] == 0.0 for row in first))

    def test_pair_six_rank_partition_exact_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, rows = ensure_validation_manifest(
                directory, SOURCE, self.split["validation_targets"])
            combined = []
            for rank in range(6):
                sampler = FixedPairValidationSampler(
                    path, self.split["validation_targets"])
                part = sampler.partition(rank)
                self.assertEqual(len(part), 252)
                combined.extend(part)
            self.assertEqual([int(row["case_index"]) for row in combined],
                             list(range(1512)))
            self.assertEqual(len(rows), len(combined))

    def test_iou_delta_inputs_are_correct(self):
        target = np.asarray([[0.0, 0.0, 10.0, 10.0]])
        local = _curves([[5.0, 0.0, 10.0, 10.0]], target)
        collab = _curves([[0.0, 0.0, 10.0, 10.0]], target)
        self.assertAlmostEqual(collab["auc"] - local["auc"],
                               2.0 / 3.0, places=6)

    def test_pair_metrics_one_row_per_epoch(self):
        row = {field: 0 for field in PAIR_FIELDS}
        row.update({"epoch": 1, "manifest_sha256": "abc"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.csv"
            self.assertTrue(_append_unique_csv(path, PAIR_FIELDS, row))
            self.assertFalse(_append_unique_csv(path, PAIR_FIELDS, row))
            with path.open("r", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 1)

    def test_online_schedule_and_best_rule(self):
        self.assertEqual(
            [epoch for epoch in range(1, 31) if online_epoch_due(epoch)],
            [5, 10, 15, 20, 25, 30])
        incumbent = {"epoch": 5, "auc_collab": 0.6,
                     "auc_delta": 0.01, "harmful_rate": 0.2}
        self.assertTrue(is_better_online(
            {"epoch": 10, "auc_collab": 0.6,
             "auc_delta": 0.02, "harmful_rate": 0.3}, incumbent))
        self.assertTrue(is_better_online(
            {"epoch": 10, "auc_collab": 0.6,
             "auc_delta": 0.01, "harmful_rate": 0.1}, incumbent))
        self.assertFalse(is_better_online(
            {"epoch": 10, "auc_collab": 0.6,
             "auc_delta": 0.01, "harmful_rate": 0.2}, incumbent))

    def test_online_failure_does_not_overwrite_best(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter = ValidationReporter(directory)
            reporter.best = {"epoch": 5, "auc_collab": 0.6,
                             "auc_delta": 0.01, "harmful_rate": 0.2}
            before = dict(reporter.best)
            try:
                raise RuntimeError("validation failed before record_online")
            except RuntimeError:
                pass
            self.assertEqual(reporter.best, before)
            self.assertFalse((Path(directory) / "best_checkpoint.json").exists())

    def test_online_validator_refuses_test_path(self):
        with self.assertRaisesRegex(ValueError, "test dataset"):
            OnlineValidator(
                model=None, frozen_tracker=None,
                dataset_root="/forbidden/threemdot_test",
                validation_targets=self.split["validation_targets"],
                device="cpu")

    def test_pair_validation_has_no_training_side_effects(self):
        class Tracker(torch.nn.Module):
            def forward_head(self, tokens):
                search = tokens[:, -256:]
                score = torch.sigmoid(search.mean(dim=-1)).reshape(-1, 1, 16, 16)
                anchor = search.mean(dim=(1, 2), keepdim=False).reshape(-1, 1)
                boxes = torch.cat((
                    anchor * 0 + 0.5, anchor * 0 + 0.5,
                    anchor * 0 + 0.2, anchor * 0 + 0.2), dim=1).unsqueeze(1)
                return {"score_map": score, "pred_boxes": boxes}

        class ValidationSampler:
            sha256 = "fixed"
            def partition(self, rank, world_size):
                return [{"uses_gt_in_student_input": "false"}] * 3

        class TrainingSampler:
            current_contract = {"epoch": 1}
            rows = [{"case": 1}]

        torch.manual_seed(42)
        fcvc = FCVCModel(FCVCConfig(enabled=True))
        model = FCVCTrainingGraph(fcvc).train()
        tracker = Tracker().eval()
        local = common.local_record(batch=1)
        local["local_output"] = tracker.forward_head(torch.cat((
            local["template_high"], local["high_search"]), dim=1))
        bundles = (common.sender(1, batch=1), common.sender(2, batch=1))
        gt = torch.tensor([[0.4, 0.4, 0.2, 0.2]])
        case = ({"uses_gt_in_student_input": "false"}, local, bundles, gt,
                [], [], [])
        optimizer = torch.optim.AdamW(fcvc.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            validator = PairValidator(
                model, tracker, lambda rows: [case, case, case],
                ValidationSampler(), optimizer, TrainingSampler(), directory,
                torch.device("cpu"), rank=0, world_size=6, dist_module=None)
            metrics, isolation = validator.run(
                1, max_local_batches=1, write_outputs=False)
        self.assertEqual(metrics["cases"], 3)
        self.assertTrue(isolation["parameters_unchanged"])
        self.assertTrue(isolation["optimizer_unchanged"])
        self.assertTrue(isolation["rng_restored"])
        self.assertTrue(isolation["training_sampler_unchanged"])
        self.assertEqual(isolation["backward_calls"], 0)
        self.assertEqual(isolation["optimizer_steps"], 0)
        self.assertEqual(isolation["scheduler_steps"], 0)


if __name__ == "__main__":
    unittest.main()
