import unittest

import torch

import common
from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel, load_fcvc_checkpoint


class FCVCCheckpointContractTest(unittest.TestCase):
    def test_old_and_new_checkpoint_schemas_load_identical_output(self):
        torch.manual_seed(42)
        source = FCVCModel(FCVCConfig(enabled=True)).eval()
        old_checkpoint = {
            "student": {k: v for k, v in source.state_dict().items() if not k.startswith("teacher.")},
            "teacher": {k: v for k, v in source.state_dict().items() if k.startswith("teacher.")},
        }
        prefixed = {"fcvc." + k: v for k, v in source.state_dict().items()}
        old_model = FCVCModel(FCVCConfig(enabled=True)).eval()
        new_model = FCVCModel(FCVCConfig(enabled=True)).eval()
        load_fcvc_checkpoint(old_model, old_checkpoint)
        load_fcvc_checkpoint(new_model, {"state_dict": prefixed})
        local = common.local_record(batch=1)
        bundles = (common.sender(1, batch=1), common.sender(2, batch=1))
        old_output = old_model(local, bundles)["reported_output"]["fcvc_search_tokens"]
        new_output = new_model(local, bundles)["reported_output"]["fcvc_search_tokens"]
        self.assertTrue(torch.equal(old_output, new_output))
