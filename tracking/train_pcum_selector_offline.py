#!/usr/bin/env python3
"""Train offline learned PCUM reliability selectors on exported CSV samples."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.pcum_selector_utils import (
    FEATURE_COLUMNS,
    compute_norm_stats,
    get_feature_columns,
    non_ignore_rows,
    normalize_features,
    read_selector_csv,
    rows_to_feature_matrix,
    validate_feature_columns,
)


class TinySelector(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def labels(rows):
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def metric_bundle(y_true, prob):
    pred = (prob >= 0.5).astype(np.int64)
    out = {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
        "prob_mean": float(np.mean(prob)),
        "prob_std": float(np.std(prob)),
        "prob_min": float(np.min(prob)),
        "prob_max": float(np.max(prob)),
    }
    try:
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="uniform")
        ece = 0.0
        for i in range(len(frac_pos)):
            ece += abs(float(frac_pos[i]) - float(mean_pred[i])) / max(len(frac_pos), 1)
        out["ece_10"] = float(ece)
    except Exception:
        out["ece_10"] = float("nan")
    return out


def train_logreg(x_train, y_train):
    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    model.fit(x_train, y_train)
    return model


def train_mlp(x_train, y_train, x_val, y_val, epochs=300, patience=30, seed=42):
    torch.manual_seed(seed)
    model = TinySelector(x_train.shape[1])
    xtr = torch.tensor(x_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    xva = torch.tensor(x_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(xva)).cpu().numpy()
        score = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.0
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_roc_auc": float(score)})
        if score > best_score + 1e-6:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_roc_auc": best_score, "history": history}


def permutation_importance(model_name, predict_prob, x_val, y_val, feature_columns):
    if len(np.unique(y_val)) <= 1:
        return []
    baseline = roc_auc_score(y_val, predict_prob(x_val))
    rng = np.random.RandomState(7)
    scores = []
    for idx, feature in enumerate(feature_columns):
        x_perm = x_val.copy()
        rng.shuffle(x_perm[:, idx])
        score = roc_auc_score(y_val, predict_prob(x_perm))
        scores.append((feature, float(baseline - score)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def write_report(path, train_rows, val_rows, metrics_by_model, logreg_weights,
                 importances_by_model, mlp_info, feature_columns):
    lines = []
    lines.append("# PCUM-v2 B1 Offline Selector Training Report")
    lines.append("")
    lines.append("## 1. 数据与泄漏控制")
    lines.append("")
    lines.append("- 使用 train 非 ignore 样本估计 feature normalization；val 仅 apply train stats。")
    lines.append("- Selector feature 白名单：`{}`。".format("`, `".join(feature_columns)))
    lines.append("- Logistic Regression 使用 `class_weight=balanced`；MLP 为 `12 -> 16 -> 1` 并使用 early stopping。")
    lines.append("")
    lines.append("| Split | Non-ignore samples | Positive | Negative |")
    lines.append("|---|---:|---:|---:|")
    for name, rows in (("train", train_rows), ("val", val_rows)):
        y = labels(rows)
        lines.append("| {} | {} | {} | {} |".format(name, len(rows), int((y == 1).sum()), int((y == 0).sum())))
    lines.append("")
    lines.append("### Full CSV Label Ratios")
    lines.append("")
    lines.append("| Split | Total rows | Usable | Positive | Negative | Ignore | Positive ratio | Negative ratio | Ignore ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    # The rows passed to this function are non-ignore only, so full split ratios
    # are written by main after loading the original CSVs.
    lines.append("__FULL_RATIO_TABLE__")
    lines.append("")

    lines.append("## 2. 分类指标（Validation, Non-ignore Frames）")
    lines.append("")
    lines.append("| Model | ROC-AUC | PR-AUC | Acc. | Precision | Recall | F1 | Brier | ECE | Prob mean/std/min/max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, metrics in metrics_by_model.items():
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}/{:.4f}/{:.4f}/{:.4f} |".format(
            name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["brier"],
            metrics["ece_10"],
            metrics["prob_mean"],
            metrics["prob_std"],
            metrics["prob_min"],
            metrics["prob_max"],
        ))
    lines.append("")

    lines.append("## 3. Logistic Regression 权重")
    lines.append("")
    lines.append("| Feature | Weight |")
    lines.append("|---|---:|")
    for feature, weight in logreg_weights:
        lines.append("| {} | {:.5f} |".format(feature, weight))
    lines.append("")

    lines.append("## 4. Permutation Importance（ROC-AUC drop）")
    lines.append("")
    model_names = list(importances_by_model.keys())
    lines.append("| Feature | {} |".format(" | ".join(model_names)))
    lines.append("|---{}|".format("|".join(["---:"] * len(model_names))))
    feature_order = [feature for feature, _ in importances_by_model.get("Logistic Regression", [])] or list(feature_columns)
    for feature in feature_order:
        values = []
        for model_name in model_names:
            values.append(dict(importances_by_model[model_name]).get(feature, 0.0))
        lines.append("| {} | {} |".format(feature, " | ".join("{:.5f}".format(v) for v in values)))
    lines.append("")
    lines.append("## 5. MLP Early Stopping")
    lines.append("")
    lines.append("- best_epoch: `{}`".format(mlp_info["best_epoch"]))
    lines.append("- best_val_roc_auc: `{:.4f}`".format(mlp_info["best_val_roc_auc"]))
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="output/pcum_v2_b1_selector/data/train_selector_samples.csv")
    parser.add_argument("--val-csv", default="output/pcum_v2_b1_selector/data/val_selector_samples.csv")
    parser.add_argument("--output-dir", default="output/pcum_v2_b1_selector")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--feature-set", choices=("base", "enhanced"), default="base")
    return parser.parse_args()


def full_label_stats(rows):
    total = len(rows)
    usable = [row for row in rows if int(row["ignore"]) == 0]
    pos = sum(1 for row in usable if int(row["label"]) == 1)
    neg = sum(1 for row in usable if int(row["label"]) == 0)
    ignore = total - len(usable)
    return {
        "total": total,
        "usable": len(usable),
        "pos": pos,
        "neg": neg,
        "ignore": ignore,
        "pos_ratio": pos / max(total, 1) * 100.0,
        "neg_ratio": neg / max(total, 1) * 100.0,
        "ignore_ratio": ignore / max(total, 1) * 100.0,
    }


def main():
    args = parse_args()
    feature_columns = get_feature_columns(args.feature_set)
    validate_feature_columns(feature_columns)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_all_rows = read_selector_csv(args.train_csv)
    val_all_rows = read_selector_csv(args.val_csv)
    train_rows = non_ignore_rows(train_all_rows)
    val_rows = non_ignore_rows(val_all_rows)
    if not train_rows or not val_rows:
        raise RuntimeError("No non-ignore selector samples available")
    x_train_raw = rows_to_feature_matrix(train_rows, feature_columns)
    y_train = labels(train_rows)
    x_val_raw = rows_to_feature_matrix(val_rows, feature_columns)
    y_val = labels(val_rows)
    stats = compute_norm_stats(x_train_raw, feature_columns)
    x_train = normalize_features(x_train_raw, stats)
    x_val = normalize_features(x_val_raw, stats)

    logreg = train_logreg(x_train, y_train)
    logreg_prob = logreg.predict_proba(x_val)[:, 1]
    logreg_metrics = metric_bundle(y_val, logreg_prob)
    logreg_weights = sorted(
        zip(feature_columns, logreg.coef_.reshape(-1).tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_prob = rf.predict_proba(x_val)[:, 1]
    rf_metrics = metric_bundle(y_val, rf_prob)

    gbdt = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        random_state=42,
    )
    gbdt.fit(x_train, y_train)
    gbdt_prob = gbdt.predict_proba(x_val)[:, 1]
    gbdt_metrics = metric_bundle(y_val, gbdt_prob)

    mlp, mlp_info = train_mlp(x_train, y_train, x_val, y_val, epochs=args.epochs, patience=args.patience)
    mlp.eval()
    with torch.no_grad():
        mlp_prob = torch.sigmoid(mlp(torch.tensor(x_val, dtype=torch.float32))).cpu().numpy()
    mlp_metrics = metric_bundle(y_val, mlp_prob)

    def logreg_predict(x):
        return logreg.predict_proba(x)[:, 1]

    def mlp_predict(x):
        with torch.no_grad():
            return torch.sigmoid(mlp(torch.tensor(x, dtype=torch.float32))).cpu().numpy()

    def rf_predict(x):
        return rf.predict_proba(x)[:, 1]

    def gbdt_predict(x):
        return gbdt.predict_proba(x)[:, 1]

    logreg_importance = permutation_importance("logreg", logreg_predict, x_val, y_val, feature_columns)
    rf_importance = permutation_importance("rf", rf_predict, x_val, y_val, feature_columns)
    gbdt_importance = permutation_importance("gbdt", gbdt_predict, x_val, y_val, feature_columns)
    mlp_importance = permutation_importance("mlp", mlp_predict, x_val, y_val, feature_columns)

    with (ckpt_dir / "feature_norm_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    with (ckpt_dir / "logreg_selector.pkl").open("wb") as handle:
        pickle.dump({"model": logreg, "feature_columns": feature_columns, "norm_stats": stats, "model_type": "logreg"}, handle)
    with (ckpt_dir / "random_forest_selector.pkl").open("wb") as handle:
        pickle.dump({"model": rf, "feature_columns": feature_columns, "norm_stats": stats, "model_type": "random_forest"}, handle)
    with (ckpt_dir / "gradient_boosting_selector.pkl").open("wb") as handle:
        pickle.dump({"model": gbdt, "feature_columns": feature_columns, "norm_stats": stats, "model_type": "gradient_boosting"}, handle)
    torch.save(
        {"state_dict": mlp.state_dict(), "feature_columns": feature_columns, "norm_stats": stats, "mlp_info": mlp_info},
        ckpt_dir / "mlp_selector.pth",
    )

    report_path = output_dir / "selector_train_report.md"
    metrics_by_model = {
        "Logistic Regression": logreg_metrics,
        "RandomForest": rf_metrics,
        "GradientBoosting": gbdt_metrics,
        "Small MLP": mlp_metrics,
    }
    importances_by_model = {
        "LogReg": logreg_importance,
        "RandomForest": rf_importance,
        "GradientBoosting": gbdt_importance,
        "MLP": mlp_importance,
    }
    write_report(report_path, train_rows, val_rows, metrics_by_model, logreg_weights,
                 importances_by_model, mlp_info, feature_columns)
    train_stats = full_label_stats(train_all_rows)
    val_stats = full_label_stats(val_all_rows)
    ratio_table = "\n".join([
        "| train | {} | {} | {} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            train_stats["total"], train_stats["usable"], train_stats["pos"], train_stats["neg"], train_stats["ignore"],
            train_stats["pos_ratio"], train_stats["neg_ratio"], train_stats["ignore_ratio"]
        ),
        "| val | {} | {} | {} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            val_stats["total"], val_stats["usable"], val_stats["pos"], val_stats["neg"], val_stats["ignore"],
            val_stats["pos_ratio"], val_stats["neg_ratio"], val_stats["ignore_ratio"]
        ),
    ])
    text = report_path.read_text(encoding="utf-8").replace("__FULL_RATIO_TABLE__", ratio_table)
    report_path.write_text(text, encoding="utf-8")
    print("Wrote {}".format(report_path))


if __name__ == "__main__":
    main()
