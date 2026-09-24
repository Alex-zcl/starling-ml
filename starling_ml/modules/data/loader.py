"""Тонкий адаптер обычного Python/PyTorch DataLoader к Context."""

from ...core.module import Module


class DataLoaderSource(Module):
    """Берёт следующий batch и раскладывает его по указанным Context keys.

    При cycle=False модуль поднимает StopIteration. Engine.iterate использует
    это стандартное поведение и не требует отдельного протокола dataloader-а.
    """

    def setup(
        self,
        loader,
        outputs,
        cycle=False,
        reset_on=None,
    ):
        self.loader = loader
        self.outputs = outputs
        self.cycle = bool(cycle)
        self.reset_on = set(reset_on or [])
        self.iterator = iter(loader)

    def __call__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            if not self.cycle:
                raise
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        if isinstance(self.outputs, dict):
            for context_key, batch_key in self.outputs.items():
                self.context[context_key] = batch[batch_key]
        else:
            for context_key, value in zip(self.outputs, batch):
                self.context[context_key] = value

    def reaction(self, signal, source=None, **payload):
        if signal in self.reset_on:
            self.iterator = iter(self.loader)
