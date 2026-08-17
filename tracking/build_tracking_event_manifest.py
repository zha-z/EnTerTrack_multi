#!/usr/bin/env python3
"""Build annotation-derived tracking event manifests without running a tracker."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from dataset_development_utils import (
        read_binary_mask,
        read_numeric_rows,
        write_csv,
    )
except ImportError:  # pragma: no cover
    from tracking.dataset_development_utils import (
        read_binary_mask,
        read_numeric_rows,
        write_csv,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_FIELDS = [
    "dataset",
    "split",
    "target_id",
    "view_id",
    "sequence_name",
    "event_source",
    "event_type",
    "start_frame",
    "end_frame",
    "frame_count",
    "threshold_or_definition",
    "peak_value",
]


def contiguous_intervals(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(flags) - 1):
            end = index if flag and index == len(flags) - 1 else index - 1
            intervals.append((start, end))
            start = None
    return intervals


def _event_row(
    metadata: dict,
    event_type: str,
    start: int,
    end: int,
    definition: str,
    peak_value: object = "",
) -> dict:
    return {
        "dataset": metadata["dataset"],
        "split": metadata["split"],
        "target_id": metadata["target_id"],
        "view_id": metadata["view_id"],
        "sequence_name": metadata["sequence_name"],
        "event_source": "annotation-derived",
        "event_type": event_type,
        "start_frame": start + 1,
        "end_frame": end + 1,
        "frame_count": end - start + 1,
        "threshold_or_definition": definition,
        "peak_value": peak_value,
    }


def compute_annotation_events(
    metadata: dict,
    boxes: Sequence[Sequence[float]],
    occlusion: Sequence[bool],
    out_of_view: Sequence[bool],
    fast_motion_threshold: float = 0.5,
    large_displacement_threshold: float = 1.0,
    scale_ratio_threshold: float = 1.5,
    long_occlusion_frames: int = 10,
    recovery_window: int = 20,
) -> List[dict]:
    """Return events derived only from GT boxes and visibility annotations."""
    frame_count = len(boxes)
    if len(occlusion) != frame_count or len(out_of_view) != frame_count:
        raise ValueError(
            "Annotation length mismatch for {}: gt={}, occlusion={}, out_of_view={}.".format(
                metadata.get("sequence_name"),
                frame_count,
                len(occlusion),
                len(out_of_view),
            )
        )
    if frame_count == 0:
        raise ValueError("No GT rows for {}.".format(metadata.get("sequence_name")))

    valid = [box[2] > 0 and box[3] > 0 for box in boxes]
    visible = [
        valid[index] and not occlusion[index] and not out_of_view[index]
        for index in range(frame_count)
    ]
    normalized_displacement = [0.0] * frame_count
    scale_change = [0.0] * frame_count
    for index in range(1, frame_count):
        previous = boxes[index - 1]
        current = boxes[index]
        if not valid[index - 1] or not valid[index]:
            continue
        previous_center = (previous[0] + previous[2] / 2.0, previous[1] + previous[3] / 2.0)
        current_center = (current[0] + current[2] / 2.0, current[1] + current[3] / 2.0)
        displacement = math.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        normalizer = max(math.sqrt(previous[2] * previous[3]), 1e-12)
        normalized_displacement[index] = displacement / normalizer
        previous_area = previous[2] * previous[3]
        current_area = current[2] * current[3]
        scale_change[index] = abs(math.log(max(current_area, 1e-12) / max(previous_area, 1e-12)))

    events: List[dict] = []
    count_types = [
        ("visible_frames", visible, "valid GT and not occluded and not out-of-view"),
        ("occlusion_frames", list(occlusion), "occlusion annotation == 1"),
        ("out_of_view_frames", list(out_of_view), "out_of_view annotation == 1"),
    ]
    for event_type, flags, definition in count_types:
        for start, end in contiguous_intervals(flags):
            events.append(_event_row(metadata, event_type, start, end, definition))

    motion_specs = [
        (
            "fast_motion_proxy",
            [value >= fast_motion_threshold for value in normalized_displacement],
            normalized_displacement,
            "GT center displacement / sqrt(previous GT area) >= {}".format(fast_motion_threshold),
        ),
        (
            "large_displacement",
            [value >= large_displacement_threshold for value in normalized_displacement],
            normalized_displacement,
            "GT center displacement / sqrt(previous GT area) >= {}".format(large_displacement_threshold),
        ),
        (
            "scale_change",
            [value >= math.log(scale_ratio_threshold) for value in scale_change],
            scale_change,
            "abs(log(GT area ratio)) >= log({})".format(scale_ratio_threshold),
        ),
    ]
    for event_type, flags, values, definition in motion_specs:
        for start, end in contiguous_intervals(flags):
            events.append(
                _event_row(
                    metadata,
                    event_type,
                    start,
                    end,
                    definition,
                    max(values[start : end + 1]),
                )
            )

    for start, end in contiguous_intervals(list(occlusion)):
        if end - start + 1 >= long_occlusion_frames:
            events.append(
                _event_row(
                    metadata,
                    "long_occlusion",
                    start,
                    end,
                    "contiguous occlusion annotation >= {} frames".format(long_occlusion_frames),
                )
            )

    invisible = [not value for value in visible]
    for start, end in contiguous_intervals(invisible):
        reappearance = end + 1
        if reappearance >= frame_count or not visible[reappearance]:
            continue
        events.append(
            _event_row(
                metadata,
                "reappearance",
                reappearance,
                reappearance,
                "first visible frame after a non-visible interval",
            )
        )
        recovery_end = min(frame_count - 1, reappearance + recovery_window - 1)
        events.append(
            _event_row(
                metadata,
                "recovery_candidate_interval",
                reappearance,
                recovery_end,
                "annotation-derived window of {} frames after reappearance".format(recovery_window),
            )
        )
    return events


def read_manifest(path: Path, dataset: str, split: str) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Sequence manifest not found: {}".format(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "split", "target_id", "view_id", "sequence_name", "sequence_path"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError("Sequence manifest is missing columns: {}".format(sorted(missing)))
    selected = [
        row
        for row in rows
        if (not dataset or row["dataset"] == dataset)
        and (not split or row["split"] == split)
    ]
    if not selected:
        raise ValueError(
            "No manifest rows match dataset={!r}, split={!r}.".format(dataset, split)
        )
    return selected


def load_tracker_events(path: Path) -> List[dict]:
    """Load optional tracker events, requiring an explicit tracker-specific label."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("event_source") != "tracker-specific":
            raise ValueError(
                "Tracker event rows must use event_source=tracker-specific; got {!r}.".format(
                    row.get("event_source")
                )
            )
    return rows


def render_statistics(rows: Sequence[dict]) -> str:
    counts: Dict[str, int] = {}
    frames: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for row in rows:
        event_type = row["event_type"]
        counts[event_type] = counts.get(event_type, 0) + 1
        frames[event_type] = frames.get(event_type, 0) + int(row.get("frame_count") or 0)
        source = row["event_source"]
        sources[source] = sources.get(source, 0) + 1
    lines = [
        "# Tracking event statistics",
        "",
        "Events marked `annotation-derived` use only GT boxes and visibility annotations. They are dataset properties and are not tracker failures. Any imported prediction-dependent event must remain labeled `tracker-specific`.",
        "",
        "| Event source | Rows |",
        "|---|---:|",
    ]
    for source, count in sorted(sources.items()):
        lines.append("| {} | {} |".format(source, count))
    lines.extend(["", "| Event type | Intervals/events | Covered frames |", "|---|---:|---:|"])
    for event_type in sorted(counts):
        lines.append("| {} | {} | {} |".format(event_type, counts[event_type], frames[event_type]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "dataset_development_audit")
    parser.add_argument("--tracker-events-csv", type=Path)
    parser.add_argument("--fast-motion-threshold", type=float, default=0.5)
    parser.add_argument("--large-displacement-threshold", type=float, default=1.0)
    parser.add_argument("--scale-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--long-occlusion-frames", type=int, default=10)
    parser.add_argument("--recovery-window", type=int, default=20)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest, args.dataset, args.split)
    event_rows: List[dict] = []
    for metadata in manifest_rows:
        sequence_path = Path(metadata["sequence_path"])
        boxes = read_numeric_rows(sequence_path / "groundtruth.txt", expected_columns=4)
        occlusion = read_binary_mask(sequence_path / "occlusion.txt", len(boxes))
        out_of_view = read_binary_mask(sequence_path / "out_of_view.txt", len(boxes))
        event_rows.extend(
            compute_annotation_events(
                metadata,
                boxes,
                occlusion,
                out_of_view,
                fast_motion_threshold=args.fast_motion_threshold,
                large_displacement_threshold=args.large_displacement_threshold,
                scale_ratio_threshold=args.scale_ratio_threshold,
                long_occlusion_frames=args.long_occlusion_frames,
                recovery_window=args.recovery_window,
            )
        )
    if args.tracker_events_csv:
        event_rows.extend(load_tracker_events(args.tracker_events_csv))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "event_manifest.csv", event_rows, EVENT_FIELDS)
    (args.output_dir / "event_statistics.md").write_text(
        render_statistics(event_rows), encoding="utf-8"
    )
    print("Wrote {} event rows to {}".format(len(event_rows), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
