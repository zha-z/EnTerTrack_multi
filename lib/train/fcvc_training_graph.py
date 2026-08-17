"""DDP-only wrapper that keeps student and teacher in one forward graph."""

import torch.nn as nn


class FCVCTrainingGraph(nn.Module):
    """Training adapter; it does not alter the FCVC model or runtime API."""

    is_fcvc_training_graph = True

    def __init__(self, fcvc):
        super().__init__()
        self.fcvc = fcvc

    def forward(self, local, sender_bundles=(), replay_fn=None,
                forward_head=None, teacher_training_payload=None):
        output = self.fcvc(
            local, sender_bundles, replay_fn=replay_fn,
            forward_head=forward_head)
        if teacher_training_payload is None:
            return output
        teacher_slots = self.fcvc.teacher(
            teacher_training_payload["mid_features"],
            teacher_training_payload["high_features"],
            teacher_training_payload["gt_roi"])
        output["teacher_slots"] = teacher_slots
        output["teacher_high"] = self.fcvc.teacher.tracking_residual(
            local["high_search"].detach(), teacher_slots)
        return output
