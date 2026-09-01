# D2-P2 Source Candidate Definition

## Frozen candidates

| candidate | requested bbox coverage | selectable | fill |
|---|---:|---|---:|
| P25 | 0.25 | yes | normalized 0.0 |
| P50 | 0.50 | yes | normalized 0.0 |
| P75 | 0.75 | yes | normalized 0.0 |
| P100 | 1.00 | historical reference only | normalized 0.0 |

No other coverage is permitted. P100 must remain exactly identical to the D1 full target-box transform.

## Frozen orientation

For each D2-P1 Clean sample, compute SHA256 over UTF-8 bytes:

```text
D2-P2-orientation-v1 + NUL + clean_sample_id
```

Interpret the first eight digest bytes as an unsigned big-endian integer and take modulo 4 over `[left, right, top, bottom]`. The orientation is therefore deterministic and shared by P25/P50/P75/P100 for the same sample.

## Frozen rasterization

The full pixel bbox is produced only by `lib/train/target_prompt_asymmetric_degradation.py::_normalized_box_to_pixels`. A candidate is a single contiguous, edge-anchored rectangle:

- left/right: full bbox height and `ceil(bbox_pixel_width * coverage)` width;
- top/bottom: full bbox width and `ceil(bbox_pixel_height * coverage)` height.

The covered axis length is clipped to `[1, full_axis_length]`. No center-box scaling, random orientation, background copy, blur, noise, color jitter, CutMix or texture replacement is allowed.

Implementation: `tracking/target_prompt_d2_p2_partial_degradation.py`. This helper is representation-only and is not imported by the training actor or sampler.
