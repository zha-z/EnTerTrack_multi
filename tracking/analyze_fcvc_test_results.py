#!/usr/bin/env python3
"""Deprecated compatibility wrapper for tracking/analysis_results.py."""

import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.fcvc_results import main


if __name__ == "__main__":
    warnings.warn(
        "analyze_fcvc_test_results.py is deprecated; use tracking/analysis_results.py",
        DeprecationWarning,
    )
    main()
