#!/usr/bin/env python3
"""Compatibility entrypoint for the opt-in FCVC scale audit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.audit_fcvc_scale import main


if __name__ == "__main__":
    main()
