"""Shared imports and fixtures for FCVC contract tests."""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
ROOT = TESTS_DIR.parent
for path in (ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_fcvc as legacy_model
import test_fcvc_runtime_integration as legacy_runtime

local_record = legacy_model.local_record
sender = legacy_model.sender
candidate = legacy_runtime.candidate
make_tracker = legacy_runtime.make_tracker
