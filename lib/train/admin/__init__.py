from .environment import env_settings, create_default_local_file_ITP_train
from .stats import AverageMeter, StatValue


class TensorboardWriter:
    def __init__(self, *args, **kwargs):
        from .tensorboard import TensorboardWriter as _TensorboardWriter

        self._writer = _TensorboardWriter(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._writer, item)
