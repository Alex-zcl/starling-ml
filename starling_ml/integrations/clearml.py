"""Опциональная явная интеграция с ClearML."""

from ..core.module import Module
from ..ops.tensors import scalar


class ClearMLLogger(Module):
    """Создаёт ClearML Task и отправляет выбранные scalars по событиям."""

    def setup(
        self,
        project_name,
        task_name,
        scalars=None,
        phase_scalars=None,
        signals=("train_step_end", "metrics_ready"),
        step_key="run.step",
        enabled=True,
        auto_connect_tensorboard=False,
    ):
        self.scalars = dict(scalars or {})
        self.phase_scalars = {
            phase: dict(values) for phase, values in (phase_scalars or {}).items()
        }
        self.signals = set(signals)
        self.step_key = step_key
        self.enabled = bool(enabled)
        self.task = None

        if self.enabled:
            try:
                from clearml import Task
            except ImportError as exc:
                raise ImportError("ClearMLLogger requires clearml") from exc

            # TensorBoard autoconnect по умолчанию выключен, потому что этот
            # модуль пишет scalars явно и иначе возможны дубликаты графиков.
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                auto_connect_frameworks={"tensorboard": auto_connect_tensorboard},
            )
            self.logger = self.task.get_logger()

    def reaction(self, signal, source=None, **payload):
        if not self.enabled:
            return
        if signal in self.signals:
            self._write(payload.get("phase"))
        if signal == "run_end" and self.task is not None:
            self.task.close()

    def _write(self, phase=None):
        """Разделяет tag `title/series` в нативный формат ClearML."""
        step = int(self.context.get(self.step_key, 0))
        selected = self.phase_scalars.get(phase, self.scalars)
        for tag, key in selected.items():
            value = self.context.get(key)
            if value is None:
                continue
            if phase in self.phase_scalars:
                tag = f"{phase}/{tag}"
            title, _, series = tag.partition("/")
            if not series:
                title, series = "scalars", title
            self.logger.report_scalar(
                title=title,
                series=series,
                value=scalar(value),
                iteration=step,
            )
