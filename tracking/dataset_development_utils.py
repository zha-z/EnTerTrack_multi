"""Shared utilities for dataset-development audits.

The helpers in this module operate on manifests and annotations only.  They do
not import or construct a tracker.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SEQUENCE_RE = re.compile(r"^(?P<target>.+)-(?P<view>[^-]+)$")


def parse_sequence_name(sequence_name: str) -> Tuple[str, str]:
    match = SEQUENCE_RE.match(sequence_name.strip())
    if not match:
        raise ValueError(
            "Sequence name {!r} must end in '-<view_id>'.".format(sequence_name)
        )
    return match.group("target"), match.group("view")


def read_sequence_list(path: Path) -> List[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Sequence list not found: {}".format(path))
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value and not value.startswith("#")]
    if not values:
        raise ValueError("Sequence list is empty: {}".format(path))
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(
            "Sequence list {} contains duplicates: {}".format(path, duplicates)
        )
    for value in values:
        parse_sequence_name(value)
    return values


def group_by_target(sequence_names: Iterable[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for sequence_name in sequence_names:
        target_id, _ = parse_sequence_name(sequence_name)
        grouped.setdefault(target_id, []).append(sequence_name)
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def split_overlap_rows(split_sequences: Mapping[str, Sequence[str]]) -> List[dict]:
    """Return sequence- and target-level overlap rows for all split pairs."""
    rows: List[dict] = []
    split_names = sorted(split_sequences)
    for index, left in enumerate(split_names):
        left_sequences = set(split_sequences[left])
        left_targets = set(group_by_target(left_sequences))
        for right in split_names[index + 1 :]:
            right_sequences = set(split_sequences[right])
            right_targets = set(group_by_target(right_sequences))
            for overlap_type, values in (
                ("sequence", sorted(left_sequences & right_sequences)),
                ("target", sorted(left_targets & right_targets)),
            ):
                rows.append(
                    {
                        "left_split": left,
                        "right_split": right,
                        "overlap_type": overlap_type,
                        "overlap_count": len(values),
                        "overlap_values": ";".join(values),
                        "status": "FAIL" if values else "PASS",
                    }
                )
    return rows


def assert_targets_not_split(split_sequences: Mapping[str, Sequence[str]]) -> None:
    owners: Dict[str, set] = {}
    for split, sequence_names in split_sequences.items():
        for target_id in group_by_target(sequence_names):
            owners.setdefault(target_id, set()).add(split)
    offenders = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
    if offenders:
        raise ValueError("Targets appear in multiple splits: {}".format(offenders))


def locate_sequence(root: Path, sequence_name: str) -> Path:
    target_id, _ = parse_sequence_name(sequence_name)
    root = Path(root)
    candidates = [
        root / target_id / sequence_name,
        root / "three" / target_id / sequence_name,
        root / "two" / target_id / sequence_name,
        root / sequence_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Sequence directory not found for {} under {} (checked {}).".format(
            sequence_name, root, ", ".join(str(path) for path in candidates)
        )
    )


def read_numeric_rows(path: Path, expected_columns: int = 0) -> List[List[float]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Annotation file not found: {}".format(path))
    rows: List[List[float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[\s,]+", line) if part]
        try:
            row = [float(part) for part in parts]
        except ValueError as error:
            raise ValueError(
                "Non-numeric annotation at {}:{}: {}".format(path, line_number, line)
            ) from error
        if expected_columns and len(row) != expected_columns:
            raise ValueError(
                "Expected {} columns at {}:{}, found {}.".format(
                    expected_columns, path, line_number, len(row)
                )
            )
        rows.append(row)
    if not rows:
        raise ValueError("Annotation file is empty: {}".format(path))
    return rows


def read_binary_mask(path: Path, expected_length: int) -> List[bool]:
    rows = read_numeric_rows(path)
    values = [bool(int(value)) for row in rows for value in row]
    if len(values) != expected_length:
        raise ValueError(
            "Mask length mismatch for {}: expected {}, found {}.".format(
                path, expected_length, len(values)
            )
        )
    return values


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(
        sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
    )
