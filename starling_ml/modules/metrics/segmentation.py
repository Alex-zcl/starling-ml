"""Накопительные segmentation-метрики для произвольной фазы."""

from ...core.module import Module
from ...ops.metrics import metrics_from_stats, multiclass_stats, multilabel_stats


class SegmentationMetrics(Module):
    """Накапливает pixel statistics и публикует их по lifecycle-сигналу.

    Один и тот же класс используется для train и validation. Различаются лишь
    входные ключи, prefix и события reset/finalize, поэтому логика не дублируется.
    """

    def setup(
        self,
        num_classes,
        prediction="model.logits",
        target="batch.mask",
        mode="multilabel",
        threshold=0.5,
        prefix="metrics.validation",
        phase="validation",
        reset_on="validation_start",
        finalize_on="validation_end",
        valid_mask=None,
        ignore_index=-100,
    ):
        self.valid_mask, self.ignore_index = valid_mask, ignore_index
        self.num_classes = int(num_classes)
        self.prediction = prediction
        self.target = target
        self.mode = mode
        self.threshold = threshold
        self.prefix = prefix
        self.phase = phase
        self.reset_on = reset_on
        self.finalize_on = finalize_on
        self.reset()

    def reset(self):
        """Удаляет накопленные counters, не меняя опубликованные метрики."""
        self.tp = self.fp = self.fn = self.tn = None

    def __call__(self):
        logits = self.context[self.prediction].detach()
        target = self.context[self.target].detach()

        mask = self.context[self.valid_mask] if self.valid_mask else None
        if self.mode in {"multilabel", "binary"}:
            stats = multilabel_stats(logits, target, self.threshold, mask, self.ignore_index)
        elif self.mode == "multiclass":
            stats = multiclass_stats(logits, target, self.num_classes, mask, self.ignore_index)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.tp is None:
            self.tp, self.fp, self.fn, self.tn = [value.clone() for value in stats]
        else:
            for current, value in zip((self.tp, self.fp, self.fn, self.tn), stats):
                current += value.to(current.device)

    def reaction(self, signal, source=None, **payload):
        # Сначала публикуем старое окно, затем reset. Это важно, когда один сигнал
        # одновременно означает конец train-окна и начало validation.
        phase_matches = payload.get("phase") in {None, self.phase}
        if (signal == self.finalize_on and phase_matches) or (signal == "run_end" and self.phase == "train"):
            self._publish()
        elif signal == self.reset_on and phase_matches:
            self.reset()

    def _publish(self):
        """Записывает tensors и mean-scalars, затем посылает metrics_ready."""
        import torch
        import torch.distributed as dist
        if dist.is_initialized():
            device = "cuda" if dist.get_backend() == "nccl" else "cpu"
            stats = torch.zeros(4, self.num_classes, device=device) if self.tp is None else torch.stack((self.tp, self.fp, self.fn, self.tn)).to(device)
            dist.all_reduce(stats)
            if stats.sum() == 0:
                return
            values = stats.unbind()
        elif self.tp is None:
            return
        else:
            values = (self.tp, self.fp, self.fn, self.tn)
        metrics = metrics_from_stats(*values)
        for name, value in metrics.items():
            self.context[f"{self.prefix}.{name}"] = value
            self.context[f"{self.prefix}.{name}_mean"] = value.mean().item()

        self.reset()
        self.signal("metrics_ready", phase=self.phase, prefix=self.prefix)

    def state_dict(self):
        # Незавершённое train-окно переживает resume без потери counters.
        return {key: getattr(self, key) for key in ("tp", "fp", "fn", "tn")}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)
