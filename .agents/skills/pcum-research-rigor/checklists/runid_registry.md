# Runid Registry

Use this registry before proposing or running validation/test. Do not overwrite existing runids.

## Fixed Test Runids

PCUM-v2A A0 weighted test on `threemdot_test`:

| Setting | Runid | Status |
|---|---:|---|
| A0 weighted raw | 18552 | final/main candidate |
| A0 weighted zero | 18554 | ablation |
| A0 weighted delay | 18555 | ablation |
| A0 weighted none | 18556 | ablation |

## Validation Runids

D0 original validation:

| Setting | Runid |
|---|---:|
| D0 T1 local-only | 19151 |
| D0 raw weighted | 19152 |
| D0 zero | 19154 |
| D0 delay | 19155 |
| D0 none | 19156 |

D0-fixed-lite ep5 validation:

| Setting | Runid |
|---|---:|
| T1 local-only | 19251 |
| raw weighted | 19252 |
| zero | 19254 |
| delay diagnostic | 19255 |
| none | 19256 |

D0-fixed-lite epoch sweep:

| Epoch | T1 | Raw |
|---:|---:|---:|
| 1 | 19311 | 19312 |
| 2 | 19321 | 19322 |
| 3 | 19331 | 19332 |
| 4 | 19341 | 19342 |

D0-fixed-lite epoch4 full ablation:

| Setting | Runid |
|---|---:|
| zero | 19344 |
| delay diagnostic | 19345 |
| none | 19346 |

D1 visible-safe epoch sweep:

| Epoch | T1 | Raw |
|---:|---:|---:|
| 1 | 19411 | 19712 |
| 2 | 19421 | 19422 |
| 3 | 19431 | 19432 |
| 4 | 19441 | 19442 |
| 5 | 19451 | 19452 |

Note: D1 epoch1 raw used clean retry runid `19712`; partial or interrupted `19412/19512` should not be used as reported results unless explicitly verified.

D1 visible-safe epoch2 full ablation validation:

| Setting | Runid |
|---|---:|
| zero | 19424 |
| delay diagnostic | 19425 |
| none | 19426 |

D1 visible-safe epoch2 full ablation clean retry validation:

| Setting | Runid | Status |
|---|---:|---|
| zero | 19434 | retry after 19424 OOM |
| delay diagnostic | 19435 | planned retry |
| none | 19436 | planned retry |

Note: D1 epoch2 zero runid `19424` logged CUDA OOM and must not be used as an accepted result.

D1 visible-safe epoch2 formal test:

| Setting | Runid |
|---|---:|
| T1 local-only | 19821 |
| raw weighted | 19822 |
| zero | 19824 |
| delay diagnostic | 19825 |
| none | 19826 |

D2-G0 remote suppression validation epoch sweep:

| Epoch | T1 local-only | Raw D2-G0 |
|---:|---:|---:|
| 1 | 19911 | 19912 |
| 2 | 19921 | 19922 |
| 3 | 19931 | 19932 |
| 4 | 19941 | 19942 |
| 5 | 19951 | 19952 |

## Registry Rules

- Check the output directory before assigning a runid.
- Prefer a new runid if any prior run was interrupted or incomplete.
- State in reports when a retry runid replaces an interrupted runid.
- Do not reuse test runids for validation or diagnostics.
- Do not reuse validation runids for final test.
