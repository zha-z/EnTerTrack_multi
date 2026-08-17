# FCVC testing

Formal command:

```bash
python tracking/test.py entertrack fcvc_full --dataset threemdot_test --threads 1 --num_gpus 1
```

This command requires the formal epoch-20 export. The refactor validation uses only synthetic/unit tests and does not invoke this command or access the test set.
