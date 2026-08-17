# FCVC checkpoint and resume

Resume with:

```bash
python tracking/train.py --script entertrack --config fcvc_full --resume /path/to/checkpoint.pth
```

The legacy resume payload and SHA256 checks remain authoritative. `load_fcvc_checkpoint` accepts split `student`/`teacher`, a `state_dict`, optional `module.` prefixes, and optional `fcvc.` prefixes without renaming model parameters.
