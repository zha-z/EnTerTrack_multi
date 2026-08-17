import tempfile
import unittest
from pathlib import Path

import torch

import common
from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel
from lib.train.data.fcvc_sampler import FCVCSampler
from lib.train.fcvc_checkpoint import load_checkpoint, save_checkpoint
from lib.train.fcvc_config import legacy_training_contract, load_resolved_config


class FakeSixRankCollective:
    @staticmethod
    def all_gather_object(outputs, value):
        for index in range(6):
            outputs[index] = value


class FCVCResumeContractTest(unittest.TestCase):
    def setUp(self):
        resolved = load_resolved_config(
            common.ROOT / "experiments/entertrack/fcvc_full.yaml")
        self.config = legacy_training_contract(resolved)
        manifest = (
            common.ROOT
            / "output/multi_agent_collaboration_clean/fcvc_manual_run/full_train_receiver_manifest.csv"
        )
        self.sampler = FCVCSampler(manifest)
        self.sampler.begin_epoch(1, rank=0, world_size=6)

    @staticmethod
    def model_optimizer():
        model = FCVCModel(FCVCConfig(enabled=True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return model, optimizer

    def test_exact_six_rank_resume_round_trip(self):
        source, source_optimizer = self.model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                path, source, source_optimizer, self.config, self.sampler,
                epoch=4, offset=300, global_step=1968, rank=0, world_size=6,
                dist_module=FakeSixRankCollective())
            target, target_optimizer = self.model_optimizer()
            state = load_checkpoint(
                path, target, target_optimizer, self.config,
                rank=3, world_size=6)
            self.assertEqual(state["current_epoch"], 4)
            self.assertEqual(state["within_epoch_case_offset"], 300)
            self.assertEqual(state["global_optimizer_step"], 1968)
            for key, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_nonzero_rank_never_writes_checkpoint(self):
        model, optimizer = self.model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.pth"
            result = save_checkpoint(
                path, model, optimizer, self.config, self.sampler,
                epoch=1, offset=3, global_step=1, rank=2, world_size=6,
                dist_module=FakeSixRankCollective())
            self.assertIsNone(result)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
