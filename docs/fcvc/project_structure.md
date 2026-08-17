# FCVC project structure

FCVC follows the OSTrack split: experiment YAMLs live in `experiments/entertrack`, orchestration in `tracking`, training responsibilities in `lib/train`, model responsibilities in `lib/models/entertrack/fcvc`, runtime tracking in `lib/test/tracker/entertrack.py`, and opt-in audits in `tools/fcvc`.

Legacy modules remain importable. Canonical names (`fcvc.py`, `query_builder.py`, `cross_view_attention.py`, `deformable_attention.py`, and `structures.py`) re-export the unchanged implementations, so state-dict keys do not change.
