from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        details = self.loss_details(prediction, target)
        return details["numerator"] / details["denominator"]

    def loss_details(self, prediction, target):
        """Return per-sample values plus the exact legacy batch reduction."""
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-12)

        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index

        reduce_dims = tuple(range(1, prediction.dim()))
        positive_sum = positive_loss.sum(dim=reduce_dims)
        negative_sum = negative_loss.sum(dim=reduce_dims)
        positive_count = positive_index.sum(dim=reduce_dims)

        per_sample_numerator = -(positive_sum + negative_sum)
        per_sample_denominator = torch.where(
            positive_count > 0,
            positive_count,
            torch.ones_like(positive_count),
        )
        total_positive = positive_count.sum()
        denominator = torch.where(
            total_positive > 0,
            total_positive,
            torch.ones_like(total_positive),
        )
        return {
            "per_sample": per_sample_numerator / per_sample_denominator,
            "per_sample_numerator": per_sample_numerator,
            "per_sample_denominator": per_sample_denominator,
            "numerator": per_sample_numerator.sum(),
            "denominator": denominator,
        }


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss
