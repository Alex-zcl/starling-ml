"""Опциональная запись Context-значений в TensorBoard."""

from ..core.module import Module
from ..ops.tensors import scalar


class TensorBoardLogger(Module):
    """Записывает scalars/images по событиям через SummaryWriter."""

    def setup(
        self,
        log_dir,
        scalars=None,
        phase_scalars=None,
        images=None,
        signals=("train_step_end", "metrics_ready"),
        step_key="run.step",
        enabled=True,
    ):
        self.scalars = dict(scalars or {})
        self.phase_scalars = {
            phase: dict(values) for phase, values in (phase_scalars or {}).items()
        }
        self.images = dict(images or {})
        self.signals = set(signals)
        self.step_key = step_key
        self.enabled = bool(enabled)
        self.writer = None

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError("TensorBoardLogger requires tensorboard") from exc
            self.writer = SummaryWriter(log_dir=log_dir)

    def reaction(self, signal, source=None, **payload):
        if not self.enabled:
            return
        if signal in self.signals:
            self._write(payload.get("phase"))
        if signal == "run_end" and self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _write(self, phase=None):
        """Логирует только значения, которые уже появились в Context."""
        step = int(self.context.get(self.step_key, 0))
        selected = self.phase_scalars.get(phase, self.scalars)
        for tag, key in selected.items():
            value = self.context.get(key)
            if value is not None:
                if phase in self.phase_scalars:
                    tag = f"{phase}/{tag}"
                self.writer.add_scalar(tag, scalar(value), step)
        for tag, key in self.images.items():
            value = self.context.get(key)
            if value is not None:
                self.writer.add_image(tag, value, step)
