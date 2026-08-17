#!/usr/bin/env python3
"""Alias for the deterministic output/loss/gradient/state audit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fcvc.identity_contract import main


if __name__ == "__main__":
    main()
