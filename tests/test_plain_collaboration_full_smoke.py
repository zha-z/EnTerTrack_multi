import copy
import os
import sys
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack import build_entertrack  # noqa: E402
from lib.train.actors.plain_collaboration import (  # noqa: E402
    build_flat_remote_tokens,
)
from lib.train.plain_collaboration_freeze import (  # noqa: E402
    apply_plain_collaboration_freeze,
)


class PlainCollaborationFullSmokeTest(unittest.TestCase):
    def test_real_plain_vit_checkpoint_forward_backward(self):
        local_cfg = copy.deepcopy(cfg)
        update_config_from_file(
            "experiments/entertrack/plain_collaboration_v1.yaml",
            base_cfg=local_cfg,
        )
        checkpoint_path = local_cfg.B0_CHECKPOINT
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(ROOT, checkpoint_path)
        self.assertTrue(os.path.isfile(checkpoint_path))
        model = build_entertrack(local_cfg, training=True)
        report = model.initialization_audit
        self.assertTrue(report["strict_full_load"])
        self.assertEqual(report["checkpoint_epoch"], 25)
        self.assertGreater(report["fresh_adapter_key_count"], 0)
        apply_plain_collaboration_freeze(model, local_cfg)
        model.train()

        torch.manual_seed(19)
        template = torch.randn(3, 3, 128, 128)
        search = torch.randn(3, 3, 256, 256)
        with torch.no_grad():
            local = model(
                template=template,
                search=search,
                return_atp=False,
                training=False,
            )
        feature = local["backbone_feat"].detach()
        self.assertEqual(feature.shape, (3, 320, 192))
        local_search = feature[:, -256:]
        remote = build_flat_remote_tokens(local_search, num_views=3).detach()
        output = model(
            template=None,
            search=None,
            collaboration_feature=feature,
            plain_remote_tokens=remote,
            plain_remote_valid=torch.ones(3, 2, dtype=torch.bool),
        )
        self.assertEqual(output["score_map"].shape, (3, 1, 16, 16))
        self.assertEqual(output["pred_boxes"].shape, (3, 1, 4))
        self.assertTrue(torch.isfinite(output["score_map"]).all())
        self.assertTrue(torch.isfinite(output["pred_boxes"]).all())

        loss = output["score_map"].mean() + output["pred_boxes"].mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        adapter_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("plain_collaboration.")
        ]
        local_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if not name.startswith("plain_collaboration.")
        ]
        self.assertTrue(any(gradient is not None for gradient in adapter_gradients))
        self.assertTrue(all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in adapter_gradients))
        self.assertTrue(all(gradient is None for gradient in local_gradients))


if __name__ == "__main__":
    unittest.main()
