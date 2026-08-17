"""Prediction-only sender message construction."""

from .sender_bundle import build_sender_bundle


class MessageEncoder:
    """Stateless callable preserving the audited sender-bundle computation."""

    def __call__(self, *args, **kwargs):
        return build_sender_bundle(*args, **kwargs)


__all__ = ["MessageEncoder", "build_sender_bundle"]
