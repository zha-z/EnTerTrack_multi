"""Safe, backward-compatible evaluation run identifiers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


RunId = Union[int, str]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
INTEGER_ARGUMENT_PATTERN = re.compile(r"^-?[0-9]+$")
IDENTITY_FILENAME = ".run_identity.json"


def validate_run_id(run_id: Optional[RunId]) -> Optional[RunId]:
    """Validate one run id without changing its model-independent meaning."""
    if run_id is None:
        return None
    if isinstance(run_id, bool):
        raise TypeError("boolean runid is not supported")
    if isinstance(run_id, int):
        return run_id
    if not isinstance(run_id, str):
        raise TypeError("runid must be an integer, string, or None")
    if not run_id:
        raise ValueError("runid must not be empty")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "invalid runid {!r}; expected {}".format(
                run_id, RUN_ID_PATTERN.pattern))
    return run_id


def parse_run_id_argument(value: str) -> RunId:
    """Preserve the historical integer CLI while accepting safe strings."""
    if value is None:
        raise ValueError("runid argument must not be None")
    text = str(value)
    if INTEGER_ARGUMENT_PATTERN.fullmatch(text):
        return int(text)
    return validate_run_id(text)


def format_run_id(run_id: RunId) -> str:
    run_id = validate_run_id(run_id)
    if isinstance(run_id, int):
        return "{:03d}".format(run_id)
    return str(run_id)


def result_directory_name(parameter_name: str,
                          run_id: Optional[RunId]) -> str:
    if run_id is None:
        return str(parameter_name)
    return "{}_{}".format(parameter_name, format_run_id(run_id))


def result_directory(results_root: Union[str, Path], tracker_name: str,
                     parameter_name: str,
                     run_id: Optional[RunId]) -> Path:
    return (
        Path(results_root)
        / str(tracker_name)
        / result_directory_name(parameter_name, run_id)
    )


def reserve_run_directory(path: Union[str, Path],
                          identity: Dict[str, Any]) -> Path:
    """Atomically reserve a new formal result directory.

    Any pre-existing directory is refused, including an interrupted run with
    the same identity. This prevents result mixing and silent overwrite.
    """
    path = Path(path)
    normalized = dict(identity)
    normalized["runid"] = validate_run_id(normalized.get("runid"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(str(path))
    except FileExistsError:
        raise FileExistsError(
            "result directory already exists; runid reuse refused: {}".format(
                path))
    marker = path / IDENTITY_FILENAME
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return marker


def read_run_identity(path: Union[str, Path]) -> Dict[str, Any]:
    marker = Path(path) / IDENTITY_FILENAME
    with marker.open("r", encoding="utf-8") as handle:
        return json.load(handle)
