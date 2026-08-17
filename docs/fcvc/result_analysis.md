# FCVC result analysis

Analyze already-generated results with:

```bash
python tracking/analysis_results.py --tracker_name entertrack --tracker_param fcvc_full --dataset_name threemdot_test
```

The unified analyzer reads existing prediction files and reports success AUC, precision, normalized precision, per-target/per-view summaries, bootstrap confidence intervals, runtime, and state-digest summaries. It does not run inference.
