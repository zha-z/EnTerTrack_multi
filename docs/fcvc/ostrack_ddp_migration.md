# FCVC OSTrack-style sampling and DDP migration

This migration changes training orchestration only. FCVC architecture, loss
definitions and weights, Safe Commit, and the frozen E0 tracker are unchanged.

| Contract | Previous formal value | Current value |
|---|---:|---:|
| Epochs | 20 | 30 |
| Receiver cases per epoch | 36132 | 10008 |
| Sync groups per epoch | full traversal | 3336 random groups |
| Global batch | 16 | 18 |
| World size | 1 | 6 |
| Micro batch per GPU | 1 | 1 |
| Accumulation steps | 16 | 3 |
| Optimizer steps per epoch | 2259 | 556 |
| Total optimizer steps | 45180 | 16680 |
| Warmup steps | old one-epoch scale | 556 |

The official-train synchronized eligibility manifest is now a legal-frame
pool. Epoch manifests are regenerated deterministically with replacement using
`epoch_seed = 42 + epoch_index`, target-balanced group selection, a maximum
template-search interval of 200, and fixed A/B/C receiver expansion. Rank 0
broadcasts the manifest contract; every rank reconstructs and verifies it,
then consumes one contiguous 556-group/1668-case partition.

No formal 30-epoch training or test-set evaluation was executed during this
migration.
