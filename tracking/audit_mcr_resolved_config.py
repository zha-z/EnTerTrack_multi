#!/usr/bin/env python3
"""Resolve one EnTeRTrack YAML and audit its declared MCR execution mode."""

import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402


class ConfigAuditError(ValueError):
    pass


def resolve_config(config_name):
    name = os.path.basename(str(config_name))
    if name.endswith(".yaml"):
        name = name[:-5]
    path = os.path.join(ROOT, "experiments", "entertrack", name + ".yaml")
    if not os.path.isfile(path):
        raise ConfigAuditError("config does not exist: {}".format(path))
    update_config_from_file(path)
    return name, cfg


def checkpoint_path(resolved):
    save_dir = str(getattr(resolved.TEST, "SAVE_DIR", ""))
    checkpoint_name = str(getattr(resolved.TEST, "CHECKPOINT_NAME", ""))
    if not checkpoint_name:
        raise ConfigAuditError("TEST.CHECKPOINT_NAME must be explicit")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ROOT, save_dir)
    return os.path.normpath(os.path.join(
        save_dir,
        "checkpoints", "train", "entertrack", checkpoint_name,
        "EnTeRTrack_ep{:04d}.pth.tar".format(int(resolved.TEST.EPOCH)),
    ))


def resolved_values(config_name):
    name, resolved = resolve_config(config_name)
    mcr = resolved.TEST.MCR
    geometry_guard = mcr.CURRENT_LARGE_SCALE_GEOMETRY_GUARD
    enabled = bool(mcr.ENABLED)
    shadow_only = bool(mcr.SHADOW_ONLY)
    mode = "DISABLED" if not enabled else ("SHADOW" if shadow_only else "ACTIVE")
    return {
        "config": name,
        "MCR.ENABLED": enabled,
        "MCR.SHADOW_ONLY": shadow_only,
        "MCR.GLOBAL_ENABLED": bool(mcr.GLOBAL_ENABLED),
        "mode": mode,
        "LOCAL_ENABLED": bool(mcr.LOCAL_ENABLED),
        "REMOTE_VERIFY_ENABLED": bool(mcr.REMOTE_VERIFY_ENABLED),
        "MULTIFRAME_CONFIRM_ENABLED": bool(mcr.MULTIFRAME_CONFIRM_ENABLED),
        "checkpoint": checkpoint_path(resolved),
        "search_size": int(resolved.TEST.SEARCH_SIZE),
        "search_factor": float(resolved.TEST.SEARCH_FACTOR),
        "local_interval": int(mcr.LOCAL_INTERVAL),
        "local_scales": list(mcr.LOCAL_SCALES),
        "confirm_frames": int(mcr.CONFIRM_FRAMES),
        "switch_margin": float(mcr.SWITCH_MARGIN),
        "guard_enabled": bool(geometry_guard.ENABLED),
        "min_scale": float(geometry_guard.MIN_SCALE),
        "min_geometry": float(geometry_guard.MIN_GEOMETRY),
    }


def audit_config(config_name, expect_mode=None):
    values = resolved_values(config_name)
    if expect_mode is not None:
        expected = str(expect_mode).upper()
        if values["mode"] != expected:
            raise ConfigAuditError(
                "expected MCR mode {}, resolved {} "
                "(ENABLED={}, SHADOW_ONLY={})".format(
                    expected,
                    values["mode"],
                    values["MCR.ENABLED"],
                    values["MCR.SHADOW_ONLY"],
                )
            )
    if values["MCR.GLOBAL_ENABLED"]:
        raise ConfigAuditError("MCR-v0 requires GLOBAL_ENABLED=false")
    return values


def _format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[{}]".format(", ".join(str(item) for item in value))
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--expect-mode", choices=("disabled", "shadow", "active"), required=True)
    args = parser.parse_args(argv)
    try:
        values = audit_config(args.config, args.expect_mode)
    except ConfigAuditError as error:
        print("MCR CONFIG AUDIT FAILED: {}".format(error), file=sys.stderr)
        return 2
    for key, value in values.items():
        print("{}={}".format(key, _format_value(value)))
    print("audit=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
