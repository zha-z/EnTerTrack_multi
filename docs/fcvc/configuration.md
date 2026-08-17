# FCVC configuration

`experiments/entertrack/fcvc_full.yaml` resolves `baseline.yaml` through `BASE_CONFIG`. The loader performs a recursive YAML update, rejects cycles, requires every FCVC model/train/data/test section, and enforces all frozen counts, losses, optimizer values, Safe Commit, seed, and epoch values. The complete resolved config is written to `<save_dir>/entertrack/fcvc_full/config.yaml`.
