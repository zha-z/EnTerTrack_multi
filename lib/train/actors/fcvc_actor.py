"""FCVC actor boundary for forward/loss/stat collection."""


class FCVCActor:
    def __init__(self, model, frozen_tracker):
        self.model = model
        self.frozen_tracker = frozen_tracker

    def __call__(self, case):
        return self.forward(case)

    def forward(self, case):
        from tracking.audit_fcvc_scale import forward_case

        losses, diagnostics = forward_case(self.model, self.frozen_tracker, case)
        stats = {
            name: float(value.detach().mean().cpu().item())
            for name, value in losses.items()
        }
        stats.update(diagnostics)
        return losses, stats
