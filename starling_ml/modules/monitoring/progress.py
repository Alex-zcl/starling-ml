"""Опциональный tqdm reporter, который ничего не меняет в обучении."""

from ...core.module import Module
from ...ops.tensors import scalar


class TqdmReporter(Module):
    """Показывает train и validation progress через lifecycle-сигналы."""

    def __init__(self, context):
        # Engine.close может быть вызван после ошибки более раннего setup-модуля.
        # Поэтому cleanup-состояние существует ещё до setup TqdmReporter.
        super().__init__(context)
        self.enabled = False
        self.ended = False
        self.train_bar = None
        self.validation_bar = None

    def setup(
        self,
        total,
        train_scalars=None,
        validation_total=None,
        validation_scalars=None,
        step_key="run.step",
        enabled=True,
        leave=True,
        mode="global",
        train_total=None,
    ):
        self.total = int(total)
        self.train_scalars = dict(train_scalars or {})
        self.validation_total = validation_total
        self.validation_scalars = dict(validation_scalars or {})
        self.step_key = step_key
        if mode not in {"global", "phase"}:
            raise ValueError("TqdmReporter mode must be global or phase")
        self.mode = mode
        self.train_total = int(train_total or total)
        import torch.distributed as dist
        self.enabled = bool(enabled) and (not dist.is_initialized() or dist.get_rank() == 0)
        self.leave = leave
        self.ended = False

        if self.enabled:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise ImportError("TqdmReporter requires tqdm") from exc
            self.tqdm = tqdm

    def reaction(self, signal, source=None, **payload):
        if not self.enabled:
            return

        if signal == "run_started" and not self.ended and self.mode == "global":
            self.train_bar = self.tqdm(total=self.total, desc="train", leave=self.leave)
        elif signal == "phase_started":
            phase = payload.get("phase")
            if phase == "train" and self.mode == "phase":
                self._close_train()
                cycle = int(payload.get("cycle", 0)) + 1
                self.train_bar = self.tqdm(total=self.train_total, desc=f"train {cycle}", leave=self.leave)
            elif phase == "validation":
                self._open_validation()
        elif signal == "phase_step_end":
            phase = payload.get("phase")
            if phase == "train":
                self._update_train(payload.get("run_step"), payload.get("step"))
            elif phase == "validation":
                self._update_validation()
        elif signal == "train_step_end":
            self._update_train()
        elif signal == "validation_start":
            if self.validation_bar is None:
                self._open_validation()
        elif signal == "validation_step_end" and self.validation_bar is not None:
            self._update_validation()
        elif signal == "validation_end":
            self._close_validation()
        elif signal == "phase_ended":
            if payload.get("phase") == "validation":
                self._close_validation()
            elif payload.get("phase") == "train" and self.mode == "phase":
                self._close_train()
        elif signal == "metrics_ready":
            if payload.get("phase") == "validation" and self.train_bar is not None:
                self.train_bar.set_postfix(self._values(self.validation_scalars))
        elif signal == "run_end":
            self.ended = True
            self._close_validation()
            if self.train_bar is not None:
                self._close_train()

    def _update_train(self, run_step=None, phase_step=None):
        """Синхронизирует bar с global step, а не с числом вызовов reporter-а."""
        if self.train_bar is None:
            return
        step = int(phase_step if self.mode == "phase" and phase_step is not None else run_step if run_step is not None else self.context[self.step_key])
        self.train_bar.update(max(step - self.train_bar.n, 0))
        self.train_bar.set_postfix(self._values(self.train_scalars))

    def _open_validation(self):
        if not self.enabled or self.validation_bar is not None:
            return
        self.validation_bar = self.tqdm(
            total=self.validation_total,
            desc="validation",
            leave=False,
        )

    def _update_validation(self):
        if self.validation_bar is None:
            return
        self.validation_bar.update(1)
        self.validation_bar.set_postfix(self._values(self.validation_scalars))

    def _values(self, mapping):
        """Читает только уже созданные scalar keys."""
        result = {}
        for name, key in mapping.items():
            value = self.context.get(key)
            if value is not None:
                result[name] = f"{scalar(value):.4f}"
        return result

    def _close_validation(self):
        """Закрывает вложенный bar безопасно при любом пути завершения."""
        if self.validation_bar is not None:
            self.validation_bar.close()
            self.validation_bar = None

    def _close_train(self):
        if self.train_bar is not None:
            self.train_bar.close()
            self.train_bar = None

    def close(self):
        self._close_validation()
        self._close_train()
