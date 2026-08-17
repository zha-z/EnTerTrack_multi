"""Online-validation logging and the single frozen best-checkpoint rule."""

import csv
import json
from pathlib import Path


ONLINE_EPOCHS = (5, 10, 15, 20, 25, 30)
ONLINE_FIELDS = (
    "epoch", "cases", "auc_local", "auc_collab", "auc_delta",
    "precision_local", "precision_collab", "precision_delta",
    "norm_precision_local", "norm_precision_collab", "norm_precision_delta",
    "helpful_rate", "harmful_rate", "tied_rate", "per_target",
    "per_view", "runtime",
)


def online_epoch_due(epoch, interval=5):
    return int(epoch) in ONLINE_EPOCHS and int(epoch) % int(interval) == 0


def is_better_online(candidate, incumbent, epsilon=1e-12):
    """AUC_collab, then AUC_delta, harmful_rate, then earlier epoch."""
    if incumbent is None:
        return True
    comparisons = (
        (candidate["auc_collab"], incumbent["auc_collab"], 1),
        (candidate["auc_delta"], incumbent["auc_delta"], 1),
        (candidate["harmful_rate"], incumbent["harmful_rate"], -1),
        (-candidate["epoch"], -incumbent["epoch"], 1),
    )
    for left, right, direction in comparisons:
        delta = (float(left) - float(right)) * direction
        if delta > epsilon:
            return True
        if delta < -epsilon:
            return False
    return False


class ValidationReporter:
    def __init__(self, output_dir, tensorboard=None):
        self.output_dir = Path(output_dir)
        self.tensorboard = tensorboard
        self.best_path = self.output_dir / "best_checkpoint.json"
        self.best = None
        if self.best_path.exists():
            self.best = json.loads(self.best_path.read_text(encoding="utf-8"))

    def record_online(self, metrics):
        epoch = int(metrics["epoch"])
        if epoch not in ONLINE_EPOCHS:
            raise ValueError("online metrics may only be written at epochs {}".format(
                ONLINE_EPOCHS))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "online_metrics.csv"
        existing = []
        has_header = path.exists() and path.stat().st_size > 0
        if has_header:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != ONLINE_FIELDS:
                    raise ValueError("unexpected online metrics schema")
                existing = list(reader)
            if any(int(row["epoch"]) == epoch for row in existing):
                return False
        row = dict(metrics)
        for field in ("per_target", "per_view", "runtime"):
            row[field] = json.dumps(row[field], sort_keys=True)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ONLINE_FIELDS)
            if not has_header:
                writer.writeheader()
            writer.writerow({field: row[field] for field in ONLINE_FIELDS})
        selected = is_better_online(metrics, self.best)
        if selected:
            self.best = {
                "epoch": epoch,
                "auc_collab": float(metrics["auc_collab"]),
                "auc_delta": float(metrics["auc_delta"]),
                "harmful_rate": float(metrics["harmful_rate"]),
                "selection_rule": [
                    "max_auc_collab", "max_auc_delta",
                    "min_harmful_rate", "earlier_epoch"],
                "checkpoint": "checkpoints/best_val_auc.pth",
                "reference_checkpoint": "checkpoints/epoch_30.pth",
            }
            self.best_path.write_text(
                json.dumps(self.best, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        if self.tensorboard is not None:
            tags = {
                "AUC_local": "auc_local", "AUC_collab": "auc_collab",
                "AUC_delta": "auc_delta", "Precision_collab": "precision_collab",
                "NormPrecision_collab": "norm_precision_collab",
                "harmful_rate": "harmful_rate",
            }
            for tag, field in tags.items():
                self.tensorboard.add_scalar(
                    "ValidationOnline/" + tag, metrics[field], epoch)
        return selected

    def write_summary(self, split_sha, pair_sha, pair_metrics=None,
                      online_metrics=None):
        lines = [
            "# FCVC validation summary", "",
            "- target split SHA256: `{}`".format(split_sha),
            "- pair manifest SHA256: `{}`".format(pair_sha),
            "- threemdot_test accessed: `false`",
            "- best rule: max AUC_collab, max AUC_delta, min harmful_rate, earlier epoch",
            "- best: `{}`".format(json.dumps(self.best, sort_keys=True)),
            "- latest pair metrics: `{}`".format(
                json.dumps(pair_metrics, sort_keys=True)),
            "- latest online metrics: `{}`".format(
                json.dumps(online_metrics, sort_keys=True)),
            "",
        ]
        path = self.output_dir / "validation_summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
