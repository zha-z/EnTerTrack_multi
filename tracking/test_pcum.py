from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd


def find_files(run_id):
    files = list(Path("output").rglob(
        f"*run{run_id}*pcum_frame_diagnostics.csv"
    ))
    if not files:
        files = [
            p for p in Path("output").rglob("*pcum_frame_diagnostics.csv")
            if str(run_id) in str(p)
        ]
    return files


def parse_box(x):
    if pd.isna(x):
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)

    text = str(x)

    for parser in (json.loads, ast.literal_eval):
        try:
            return np.asarray(parser(text), dtype=float)
        except Exception:
            pass

    return None


def bbox_stats(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns:
        return None

    equal_count = 0
    valid_count = 0
    diffs = []

    for x, y in zip(df[col1], df[col2]):
        a = parse_box(x)
        b = parse_box(y)

        if a is None or b is None or a.shape != b.shape:
            continue

        valid_count += 1
        diff = np.abs(a - b)
        diffs.append(diff)

        if np.allclose(a, b, atol=1e-8, rtol=0):
            equal_count += 1

    if not diffs:
        return None

    diffs = np.stack(diffs)

    return {
        "valid": valid_count,
        "equal_ratio": equal_count / valid_count,
        "mean_abs_diff": float(diffs.mean()),
        "max_abs_diff": float(diffs.max()),
        "per_coord_mean": diffs.mean(axis=0).tolist(),
        "per_coord_max": diffs.max(axis=0).tolist(),
    }


def load_run(run_id):
    files = find_files(run_id)

    print("\n" + "=" * 70)
    print("RUN:", run_id)
    print("CSV files:", len(files))

    if not files:
        return None

    frames = []

    for path in files:
        try:
            df = pd.read_csv(path)
            df["_source_file"] = str(path)
            frames.append(df)
        except Exception as exc:
            print("读取失败:", path, exc)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    print("Total rows:", len(df))
    print("Columns:", len(df.columns))

    for col in ["current_uav", "uav_id", "uav"]:
        if col in df:
            print("\nUAV distribution:")
            print(df[col].value_counts(dropna=False))
            break

    if "remote_uav_count" in df:
        remote_count = pd.to_numeric(
            df["remote_uav_count"], errors="coerce"
        )

        print("\nRemote UAV count:")
        print(remote_count.value_counts(dropna=False).sort_index())
        print(
            "Remote participation ratio:",
            float((remote_count.fillna(0) > 0).mean()),
        )

    pairs = [
        ("local_bbox", "raw_collaborative_bbox"),
        ("raw_collaborative_bbox", "final_bbox"),
        ("local_bbox", "final_bbox"),
        ("predicted_bbox_local", "predicted_bbox_collaborative"),
    ]

    for col1, col2 in pairs:
        result = bbox_stats(df, col1, col2)
        if result is not None:
            print(f"\n{col1} VS {col2}")
            for key, value in result.items():
                print(f"{key}: {value}")

    for col in [
        "instant_delta_iou",
        "final_delta_iou",
        "fallback_delta_iou",
    ]:
        if col in df:
            x = pd.to_numeric(df[col], errors="coerce").dropna()

            print(f"\n{col}:")
            print(x.describe())
            print("abs > 1e-8:", float((x.abs() > 1e-8).mean()))
            print("abs > 1e-5:", float((x.abs() > 1e-5).mean()))
            print("abs > 1e-4:", float((x.abs() > 1e-4).mean()))
            print("abs > 1e-3:", float((x.abs() > 1e-3).mean()))

    for col in [
        "prompt_norm",
        "aligned_prompt_norm",
        "alignment_gate_mean",
        "alignment_gate_std",
        "fusion_gate_mean",
    ]:
        if col in df:
            x = pd.to_numeric(df[col], errors="coerce")

            print(f"\n{col}:")
            print(x.describe())
            print("non-NaN:", int(x.notna().sum()))
            print("non-zero:", int((x.fillna(0).abs() > 1e-8).sum()))

    if "fallback_triggered" in df:
        print("\nFallback triggered:")
        print(df["fallback_triggered"].value_counts(dropna=False))

    if "final_source" in df:
        print("\nFinal source:")
        print(df["final_source"].value_counts(dropna=False))

    return df


run301 = load_run(301)
run302 = load_run(302)