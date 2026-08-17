class EasyDict(dict):
    """Small EasyDict-compatible fallback used when the package is absent."""

    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        if mapping is not None:
            self.update(mapping)
        if kwargs:
            self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, mapping=None, **kwargs):
        items = []
        if mapping is not None:
            items.extend(dict(mapping).items())
        items.extend(kwargs.items())
        for key, value in items:
            super().__setitem__(key, self._convert(value))

    def copy(self):
        return EasyDict(self)

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return EasyDict(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value
