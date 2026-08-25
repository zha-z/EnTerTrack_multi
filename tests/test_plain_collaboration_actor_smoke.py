import copy
import os
import sys
import unittest
from types import SimpleNamespace

import torch
from torch import nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg  # noqa: E402
from lib.models.entertrack.entertrack import EnTeRTrack  # noqa: E402
from lib.models.entertrack.plain_collaboration import (  # noqa: E402
    PlainCollaborationV1,
)
from lib.train.actors.entertrack_threemdot import (  # noqa: E402
    EnTeRTrackActorThreeMDOT,
)


class FakePlainBackbone(nn.Module):
    def __init__(self, token_dim=12):
        super().__init__()
        self.token_dim = token_dim
        self.projection = nn.Linear(1, token_dim)

    def forward(self, z, x, **kwargs):
        value = x.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
        token = self.projection(value).unsqueeze(1)
        return token.expand(-1, 320, -1), {}


class FakeCenterHead(nn.Module):
    feat_sz = 16

    def forward(self, feature, gt_score_map=None):
        batch_size = feature.shape[0]
        score = feature.mean(dim=1, keepdim=True)
        bbox = feature.new_zeros(batch_size, 1, 4)
        size = feature.new_zeros(batch_size, 2, 16, 16)
        offset = feature.new_zeros(batch_size, 2, 16, 16)
        return score, bbox, size, offset


class PlainCollaborationActorSmokeTest(unittest.TestCase):
    def test_flat_abc_dispatch_and_backward(self):
        local_cfg = copy.deepcopy(cfg)
        local_cfg.MODEL.PLAIN_COLLABORATION.ENABLED = True
        local_cfg.TRAIN.PLAIN_COLLABORATION.ENABLED = True
        local_cfg.TRAIN.PLAIN_COLLABORATION.DETACH_REMOTE = True
        local_cfg.TRAIN.MULTIVIEW.REQUIRE_ALL_VIEWS_VISIBLE = True
        local_cfg.TRAIN.MULTIVIEW.CANONICAL_VIEW_ORDER = True
        local_cfg.TRAIN.MULTIVIEW.DIAGNOSTICS_ENABLED = False
        local_cfg.MODEL.BACKBONE.CE_LOC = []

        model = EnTeRTrack(
            transformer=FakePlainBackbone(),
            box_head=FakeCenterHead(),
            head_type="CENTER",
            plain_collaboration=PlainCollaborationV1(
                token_dim=12, num_heads=3, enabled=True),
            plain_collaboration_freeze_local=True,
        )
        actor = EnTeRTrackActorThreeMDOT(
            net=model,
            objective={},
            loss_weight={},
            settings=SimpleNamespace(batchsize=1),
            cfg=local_cfg,
        )
        data = {
            "template_images": torch.randn(3, 1, 3, 8, 8),
            "search_images": torch.randn(3, 1, 3, 8, 8),
            "template_anno": torch.tensor([[[0.2, 0.2, 0.3, 0.3]]] * 3),
            "search_anno": torch.tensor([[[0.2, 0.2, 0.3, 0.3]]] * 3),
            "target_id": ["T0"],
            "view_ids": [["A"], ["B"], ["C"]],
            "search_frame_ids": [[11], [11], [11]],
            "epoch": 0,
        }
        output = actor.forward_pass(model, data)
        self.assertTrue(output["pcum_flat_multiview"])
        self.assertEqual(output["num_views"], 3)
        self.assertEqual(output["score_map"].shape, (3, 1, 16, 16))
        self.assertEqual(
            output["plain_collaboration"]["valid_remote_count"].tolist(),
            [2, 2, 2],
        )
        loss = output["score_map"].mean()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.plain_collaboration.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
