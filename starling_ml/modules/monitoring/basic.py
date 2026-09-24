"""Простой консольный reporter без внешних зависимостей."""

from ...core.module import Module
from ...ops.tensors import scalar


class ConsoleLogger(Module):
    """Печатает выбранные scalar keys на заданных событиях."""

    def setup(
        self,
        scalars,
        signals=("train_step_end", "metrics_ready", "run_end"),
        every=1,
        step_key="run.step",
    ):
        import torch.distributed as dist
        self.enabled = not dist.is_initialized() or dist.get_rank() == 0
        self.scalars = dict(scalars)
        self.signals = set(signals)
        self.every = max(int(every), 1)
        self.step_key = step_key

    def reaction(self, signal, source=None, **payload):
        if not self.enabled or signal not in self.signals:
            return

        step = int(self.context.get(self.step_key, 0))
        if signal == "train_step_end" and step % self.every != 0:
            return

        values = []
        for name, key in self.scalars.items():
            value = self.context.get(key)
            if value is not None:
                values.append(f"{name}={scalar(value):.4f}")

        phase = payload.get("phase")
        header = f"step={step}"
        if phase:
            header += f" phase={phase}"
        if values:
            print(header + " " + " ".join(values))
