import torch


class RemotePromptAblator:
    """Apply deterministic prompt-level remote ablations during evaluation."""

    MODES = ("normal", "zero", "temporal_shuffle")

    def __init__(self, mode="normal", offset=10):
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(
                "Unsupported PCUM remote ablation mode: {}".format(mode)
            )
        self.offset = int(offset)
        if self.offset <= 0:
            raise ValueError("PCUM remote ablation offset must be positive")
        self.reset()

    def reset(self):
        self._histories = None

    def record(self, prompts):
        """Record one prompt per source UAV for the current frame."""
        if self.mode != "temporal_shuffle":
            return

        if self._histories is None:
            self._histories = [[] for _ in prompts]
        elif len(self._histories) != len(prompts):
            raise ValueError("The number of remote prompt sources changed")

        max_history = self.offset + 1
        for source_index, prompt in enumerate(prompts):
            stored = None
            if prompt is not None:
                stored = prompt.detach().to(device="cpu").clone()
            history = self._histories[source_index]
            history.append(stored)
            if len(history) > max_history:
                del history[0]

    def apply(self, source_index, prompt, target_device=None):
        """Return an ablated prompt while preserving shape, dtype, and device."""
        current = prompt.detach()
        if target_device is not None:
            current = current.to(device=target_device)

        if self.mode == "normal":
            return current
        if self.mode == "zero":
            return torch.zeros_like(current)

        if self._histories is None:
            raise RuntimeError("Temporal prompt history has not been recorded")
        if source_index < 0 or source_index >= len(self._histories):
            raise IndexError("Remote prompt source index is out of range")

        history = self._histories[source_index]
        if not history:
            raise RuntimeError("Temporal prompt history is empty")

        # After recording frame t, index t-10 is len(history)-1-offset.
        # During warm-up this clamps to the earliest available prompt.
        history_index = max(0, len(history) - 1 - self.offset)
        delayed = history[history_index]
        if delayed is None:
            delayed = current.to(device="cpu")
        return delayed.to(device=current.device, dtype=current.dtype)
