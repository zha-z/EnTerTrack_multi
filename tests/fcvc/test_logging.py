import csv
import io
import math
import tempfile
import unittest
from pathlib import Path

import torch

import common
from lib.train.fcvc_logging import (
    CASE_METRIC_FIELDS,
    EPOCH_FIELDS,
    STEP_FIELDS,
    FCVCStepLogger,
    MetricsCSVWriter,
    extract_case_metrics,
    format_console_line,
    logical_batch_case_count,
    should_print_step,
)
from tracking.analysis_training import analyze as analyze_training


def synthetic_losses():
    values = {
        "L_cls": 0.5,
        "L_giou": 0.25,
        "L_l1": 0.1,
        "L_track": 1.5,
        "L_align": 0.4,
        "L_recon": 0.7,
        "L_safe": 0.2,
        "L_cycle": 0.3,
        "L_teacher_track": 0.8,
    }
    total = 1.5 + 0.5 * 0.4 + 0.7 + 0.5 * 0.2 + 0.1 * 0.3 + 0.25 * 0.8
    values["L_total"] = total
    return {
        key: torch.tensor([value], requires_grad=True)
        for key, value in values.items()
    }


def synthetic_diagnostics():
    return {
        "collaborative_mean_iou": torch.tensor(0.4382, requires_grad=True),
        "residual_local_feature_norm_ratio": 0.0849,
        "global_matcher_null_attention_ratio": 0.2115,
        "sender_1_attention_contribution": 0.3,
        "sender_2_attention_contribution": 0.4,
    }


def step_row(epoch, step, global_step, cases, metric_value=1.0):
    row = {
        "epoch": epoch,
        "step_in_epoch": step,
        "global_step": global_step,
        "processed_cases": cases,
        "lr": 2.21e-6,
        "fps_interval": 0.0,
        "fps_epoch": 0.0,
        "data_time": 0.035,
        "forward_time": 0.05,
        "total_time": 0.1,
        "grad_norm_before_clip": 2.0,
        "grad_norm_after_clip": 1.0,
        "clip_applied": 1.0,
        "gpu_memory_allocated": 100.0,
        "gpu_memory_reserved": 200.0,
    }
    for field in CASE_METRIC_FIELDS:
        row[field] = float(metric_value)
    return row


class FCVCLoggingContractTest(unittest.TestCase):
    def test_raw_weighted_loss_semantics_and_total_sum(self):
        metrics = extract_case_metrics(synthetic_losses(), synthetic_diagnostics())
        self.assertAlmostEqual(metrics["loss_track"], 1.5, places=6)
        self.assertAlmostEqual(metrics["loss_giou"], 0.25, places=6)
        self.assertAlmostEqual(metrics["loss_l1"], 0.1, places=6)
        self.assertAlmostEqual(metrics["loss_align_raw"], 0.4, places=6)
        self.assertAlmostEqual(metrics["loss_align_weighted"], 0.2, places=6)
        self.assertAlmostEqual(metrics["loss_recon_weighted"], 0.7, places=6)
        self.assertAlmostEqual(metrics["loss_safe_weighted"], 0.1, places=6)
        self.assertAlmostEqual(metrics["loss_cycle_weighted"], 0.03, places=6)
        self.assertAlmostEqual(metrics["loss_teacher_weighted"], 0.2, places=6)
        component_sum = sum(metrics[name] for name in (
            "loss_track", "loss_align_weighted", "loss_recon_weighted",
            "loss_safe_weighted", "loss_cycle_weighted", "loss_teacher_weighted"))
        self.assertAlmostEqual(metrics["loss_total"], component_sum, places=6)
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value)
                            for value in metrics.values()))

    def test_iou_is_detached_finite_python_float(self):
        diagnostics = synthetic_diagnostics()
        self.assertTrue(diagnostics["collaborative_mean_iou"].requires_grad)
        metrics = extract_case_metrics(synthetic_losses(), diagnostics)
        self.assertIsInstance(metrics["iou"], float)
        self.assertFalse(hasattr(metrics["iou"], "requires_grad"))
        self.assertTrue(math.isfinite(metrics["iou"]))

    def test_console_format_matches_optimizer_step_contract(self):
        values = step_row(1, 50, 50, 16)
        values.update({
            "fps": 15.2, "loss_total": 4.2089, "loss_track": 1.6714,
            "loss_giou": 0.5489, "loss_l1": 0.1086,
            "loss_align_weighted": 0.1778, "loss_recon_weighted": 1.5512,
            "loss_safe_weighted": 0.47715, "loss_cycle_weighted": 0.09286,
            "loss_teacher_weighted": 0.23848, "iou": 0.4382,
            "safe_active_rate": 0.92448, "residual_local_ratio": 0.0849,
            "null_attention": 0.2115,
        })
        epoch_values = dict(values)
        epoch_values["fps"] = 20.0
        line = format_console_line(1, 50, 2259, values, epoch_values)
        self.assertTrue(line.startswith("[train: 1, 50 / 2259] FPS: 15.2 (20.0)"))
        self.assertIn("Loss/total: 4.20890", line)
        self.assertIn("Loss/align_w: 0.17780", line)
        self.assertIn("IoU: 0.43820", line)
        self.assertNotIn("microbatch", line.lower())

    def test_print_frequency_is_first_every_50_and_last(self):
        printed = [
            step for step in range(1, 2260)
            if should_print_step(step, 2259, print_interval=50)
        ]
        self.assertEqual(printed[0], 1)
        self.assertEqual(printed[-1], 2259)
        self.assertEqual(printed[1:-1], list(range(50, 2251, 50)))

    def test_800_microbatches_produce_50_step_rows_and_two_console_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            logger = FCVCStepLogger(directory, steps_per_epoch=2259, stream=stream)
            logger.begin_epoch(1)
            optimizer_steps = 0
            for microbatch in range(1, 801):
                if microbatch % 16 != 0:
                    continue
                optimizer_steps += 1
                logger.record_step(step_row(
                    1, optimizer_steps, optimizer_steps, 16))
            with (Path(directory) / "step_metrics.csv").open(
                    "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            lines = [line for line in stream.getvalue().splitlines() if line]
            self.assertEqual(len(rows), 50)
            self.assertEqual(len(lines), 2)
            self.assertIn("[train: 1, 1 / 2259]", lines[0])
            self.assertIn("[train: 1, 50 / 2259]", lines[1])

    def test_last_logical_batch_uses_four_cases_and_divisor_four(self):
        self.assertEqual(36132 % 16, 4)
        self.assertEqual(logical_batch_case_count(36128, 36132, 16), 4)
        last_batch_reduced_loss = sum(1.0 / 4.0 for _ in range(4))
        full_batch_reduced_loss = sum(1.0 / 16.0 for _ in range(16))
        self.assertEqual(last_batch_reduced_loss, 1.0)
        self.assertEqual(full_batch_reduced_loss, 1.0)

    def test_step_csv_schema_header_once_and_resume_has_no_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            first = FCVCStepLogger(directory, steps_per_epoch=2259, stream=io.StringIO())
            first.begin_epoch(1)
            first.record_step(step_row(1, 1, 1, 16))
            first.record_step(step_row(1, 2, 2, 16))
            resumed = FCVCStepLogger(directory, steps_per_epoch=2259, stream=io.StringIO())
            resumed.begin_epoch(1)
            self.assertIsNone(resumed.record_step(step_row(1, 2, 2, 16)))
            resumed.record_step(step_row(1, 3, 3, 16))
            path = Path(directory) / "step_metrics.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), STEP_FIELDS)
            self.assertEqual(lines.count(",".join(STEP_FIELDS)), 1)
            self.assertEqual([int(row["global_step"]) for row in rows], [1, 2, 3])

    def test_epoch_csv_is_case_weighted_and_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = FCVCStepLogger(directory, steps_per_epoch=2, stream=io.StringIO())
            logger.begin_epoch(1)
            logger.record_step(step_row(1, 1, 1, 16, metric_value=1.0))
            logger.record_step(step_row(1, 2, 2, 4, metric_value=5.0))
            self.assertTrue(logger.finish_epoch(global_step=2))
            self.assertFalse(logger.finish_epoch(global_step=2))
            path = Path(directory) / "epoch_metrics.csv"
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), EPOCH_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["processed_cases"]), 20)
            self.assertAlmostEqual(float(rows[0]["loss_total"]), 1.8, places=6)

    def test_training_analysis_reads_only_csv_and_writes_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "entertrack" / "fcvc_full"
            logger = FCVCStepLogger(run_dir, steps_per_epoch=2, stream=io.StringIO())
            logger.begin_epoch(1)
            logger.record_step(step_row(1, 1, 1, 16))
            logger.record_step(step_row(1, 2, 2, 4))
            logger.finish_epoch(global_step=2)
            summary = analyze_training("entertrack", "fcvc_full", directory)
            self.assertEqual(summary["optimizer_steps"], 2)
            self.assertFalse(summary["model_loaded"])
            self.assertFalse(summary["image_loaded"])
            self.assertFalse(summary["test_set_accessed"])
            self.assertEqual(len(summary["generated"]), 5)
            self.assertTrue(all(Path(path).is_file()
                                for path in summary["generated"]))


if __name__ == "__main__":
    unittest.main()
