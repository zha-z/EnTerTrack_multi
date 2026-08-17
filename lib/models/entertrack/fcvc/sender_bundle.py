from typing import Optional, Tuple

import torch

from .structures import SenderBundle


_FLOAT_FIELDS = (
    "mid_features", "high_features", "response_map", "confidence_uncertainty",
    "target_prototype", "position_grid", "crop_affine", "local_bbox",
)


def normalized_position_grid(batch: int, grid_size: int, device, dtype) -> torch.Tensor:
    coords = torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size,
                            grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
    return grid.repeat(batch, 1, 1, 1)


def _entropy(response: torch.Tensor) -> torch.Tensor:
    p = response.clamp(1e-6, 1.0 - 1e-6)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / 0.6931471805599453


def _weighted_prototype(high_features: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    b, n, c = high_features.shape
    weight = response.reshape(b, 1, n).transpose(1, 2).to(dtype=high_features.dtype)
    denom = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (high_features * weight).sum(dim=1) / denom.squeeze(1)


def build_sender_bundle(
    mid_features: torch.Tensor,
    high_features: torch.Tensor,
    response_map: torch.Tensor,
    crop_affine: Optional[torch.Tensor] = None,
    image_size: Optional[torch.Tensor] = None,
    local_bbox: Optional[torch.Tensor] = None,
    view_id: Optional[torch.Tensor] = None,
    timestamp: Optional[torch.Tensor] = None,
    dense_dtype: torch.dtype = torch.float16,
) -> SenderBundle:
    if mid_features.shape != high_features.shape:
        raise ValueError("mid/high feature shapes must match")
    b, n, c = high_features.shape
    grid = int(n ** 0.5)
    if grid * grid != n:
        raise ValueError("sender features must form a square grid")
    device = high_features.device
    response = response_map.detach().to(device=device, dtype=dense_dtype)
    conf_unc = torch.cat((response, _entropy(response).to(dtype=dense_dtype)), dim=1)
    high_detached = high_features.detach().to(dtype=dense_dtype)
    proto = _weighted_prototype(high_detached, response).detach().to(dtype=dense_dtype)
    pos = normalized_position_grid(b, grid, device, dense_dtype).detach()
    if crop_affine is None:
        crop_affine = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
    if image_size is None:
        image_size = torch.full((b, 2), grid * 16, device=device, dtype=torch.int32)
    if local_bbox is None:
        local_bbox = torch.zeros(b, 4, device=device)
    if view_id is None:
        view_id = torch.zeros(b, device=device, dtype=torch.int16)
    if timestamp is None:
        timestamp = torch.zeros(b, device=device, dtype=torch.int64)
    return SenderBundle(
        mid_features=mid_features.detach().to(dtype=dense_dtype),
        high_features=high_detached,
        response_map=response,
        confidence_uncertainty=conf_unc.detach(),
        target_prototype=proto,
        position_grid=pos,
        crop_affine=crop_affine.detach().to(dtype=torch.float32),
        image_size=image_size.detach().to(dtype=torch.int32),
        local_bbox=local_bbox.detach().to(dtype=torch.float32),
        view_id=view_id.detach().to(dtype=torch.int16),
        timestamp=timestamp.detach().to(dtype=torch.int64),
    )


def validate_sender_pair(
    bundles: Tuple[SenderBundle, ...],
    batch: int,
    token_dim: int = 192,
    search_tokens: int = 256,
) -> Tuple[bool, str]:
    if len(bundles) != 2:
        return False, "requires exactly two sender bundles"
    seen = set()
    for bundle in bundles:
        schema = bundle.schema()
        if tuple(bundle.mid_features.shape) != (batch, search_tokens, token_dim):
            return False, "invalid mid feature shape"
        if tuple(bundle.high_features.shape) != (batch, search_tokens, token_dim):
            return False, "invalid high feature shape"
        if not torch.isfinite(bundle.crop_affine).all():
            return False, "nonfinite crop affine"
        det = torch.det(bundle.crop_affine)
        if not torch.isfinite(det).all() or (det.abs() < 1e-8).any():
            return False, "singular crop affine"
        for name in _FLOAT_FIELDS:
            if not torch.isfinite(getattr(bundle, name)).all():
                return False, "nonfinite {}".format(name)
        key = tuple(int(v) for v in bundle.view_id.cpu().reshape(-1).tolist())
        if key in seen:
            return False, "duplicate sender view"
        seen.add(key)
        if not all(item["detached"] for item in schema.values()):
            return False, "bundle tensors must be detached"
    if not torch.equal(bundles[0].timestamp, bundles[1].timestamp):
        return False, "timestamp mismatch"
    return True, "ok"
