# FCVC training

Formal command:

```bash
python tracking/train.py --script entertrack --config fcvc_full --save_dir ./output --mode single
```

Use `--dry-run` for config, manifest, schedule, and resume-contract validation without model construction or optimizer steps. The fixed contract is seed 42, 20 epochs, 36,132 receiver cases, 2,259 optimizer steps per epoch, and 45,180 total steps. Formal training is not run as part of the refactor.
