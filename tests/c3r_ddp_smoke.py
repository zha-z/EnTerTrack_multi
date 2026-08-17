"""Two-rank synthetic DDP consistency smoke for C3R modules only."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lib.models.entertrack.c3r import C3R


class C3RDDPSmoke(nn.Module):
    def __init__(self):
        super().__init__()
        self.c3r = C3R(variant="c1")

    def forward(self, local_tokens, local_response, remote_tokens, remote_response):
        remote_prompt = self.c3r.encoder.pool_project(remote_tokens, remote_response)
        peer_output = self.c3r.adapter(local_tokens, remote_prompt)
        reliability_input = local_tokens.new_tensor(
            [[0.8, 0.7, 0.8, 0.2, 0.75, 0.65, 0.7, 0.15, 0.4, 0.25]])
        gate, logit = self.c3r.reliability(reliability_input)
        fused, diagnostics = self.c3r.fusion(local_tokens, [peer_output], [gate])
        return fused, logit, diagnostics["residual_budget_loss"]


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260716)
    model = DistributedDataParallel(
        C3RDDPSmoke().to(device), device_ids=[local_rank], broadcast_buffers=False)
    torch.manual_seed(20260717)
    local_tokens = torch.randn(1, 256, 192, device=device)
    local_response = torch.sigmoid(torch.randn(1, 1, 16, 16, device=device))
    remote_tokens = torch.randn(1, 256, 192, device=device)
    remote_response = torch.sigmoid(torch.randn(1, 1, 16, 16, device=device))
    fused, logit, budget = model(
        local_tokens, local_response, remote_tokens, remote_response)
    loss = fused.square().mean() + 0.10 * logit.square().mean() + 0.05 * budget
    loss.backward()
    output_checksum = torch.stack((fused.sum(), loss.detach()))
    gradient_checksum = torch.stack([
        parameter.grad.float().sum()
        for parameter in model.parameters() if parameter.grad is not None
    ]).sum().reshape(1)
    gathered_outputs = [torch.zeros_like(output_checksum) for _ in range(dist.get_world_size())]
    gathered_gradients = [torch.zeros_like(gradient_checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_outputs, output_checksum)
    dist.all_gather(gathered_gradients, gradient_checksum)
    reference_output = gathered_outputs[0]
    reference_gradient = gathered_gradients[0]
    if not all(torch.equal(value, reference_output) for value in gathered_outputs[1:]):
        raise RuntimeError("C3R DDP output inconsistency")
    if not all(torch.equal(value, reference_gradient) for value in gathered_gradients[1:]):
        raise RuntimeError("C3R DDP gradient inconsistency")
    if dist.get_rank() == 0:
        print("C3R_DDP_SMOKE_PASS world_size={} output_checksum={} gradient_checksum={}".format(
            dist.get_world_size(), reference_output.tolist(), reference_gradient.item()))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
