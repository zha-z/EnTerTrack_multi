import unittest

import common
from lib.train.fcvc_config import legacy_training_contract, load_resolved_config


class FCVCTrainingContractTest(unittest.TestCase):
    test_sender_bundle_is_prediction_only = common.legacy_model.FCVCContractTest.test_sender_bundle_schema_and_no_gt

    def test_resolved_yaml_matches_ostrack_ddp_training_contract(self):
        resolved = load_resolved_config(
            common.ROOT / "experiments/entertrack/fcvc_full.yaml")
        mapped = legacy_training_contract(resolved)
        self.assertEqual(mapped["world_size"], 6)
        self.assertEqual(mapped["global_batch_size"], 18)
        self.assertEqual(mapped["microbatch_size"], 1)
        self.assertEqual(mapped["gradient_accumulation_steps"], 3)
        self.assertEqual(mapped["sample_per_epoch"], 10008)
        self.assertEqual(mapped["steps_per_epoch"], 556)
        self.assertEqual(mapped["total_steps"], 16680)
        self.assertEqual(resolved["TRAIN"]["EPOCH"], 30)
        self.assertEqual(resolved["DATA"]["TRAIN"]["SYNC_GROUPS_PER_EPOCH"], 3336)
        self.assertTrue(mapped["data_contract"]["replacement_sampling"])
        self.assertTrue(mapped["data_contract"]["target_balanced"])
