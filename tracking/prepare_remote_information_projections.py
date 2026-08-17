"""Create preregistered fixed orthonormal projections without reading labels."""

from pathlib import Path
import hashlib
import json

import numpy as np


SEED = 20260719
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/multi_agent_collaboration_clean/remote_information_sufficiency"


def orthonormal(rng, rows, cols):
    matrix = rng.standard_normal((rows, cols))
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return (q * signs).astype(np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "fixed_projections.npz"
    if path.exists():
        raise FileExistsError(str(path))
    rng = np.random.default_rng(SEED)
    arrays = {
        "prompt_local_256x16": orthonormal(rng, 256, 16),
        "prompt_remote_256x16": orthonormal(rng, 256, 16),
        "prompt_difference_256x16": orthonormal(rng, 256, 16),
        "prompt_product_256x16": orthonormal(rng, 256, 16),
        "residual_channel_mean_std_384x32": orthonormal(rng, 384, 32),
    }
    np.savez(path, **arrays)
    manifest = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "seed": SEED,
        "utility_or_gt_read": False,
        "matrices": {name: list(value.shape) for name, value in arrays.items()},
    }
    manifest_path = OUTPUT / "fixed_projections_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
