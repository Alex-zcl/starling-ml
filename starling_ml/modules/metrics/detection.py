"""Detection-метрики, полученные из segmentation masks."""

import torch

from ...core.module import Module


class SegmentationDetectionMetrics(Module):
    """Оценивает presence/detection по площади пересечения масок.

    Модуль не пытается стать полноценной object-detection библиотекой. Он
    сохраняет полезную старую механику для задач, где объект представлен маской.
    """

    def setup(
        self,
        num_classes,
        average_object_size,
        prediction="model.logits",
        target="batch.mask",
        threshold=0.5,
        false_positive_fraction=0.25,
        false_negative_fraction=0.25,
        prefix="detection.validation",
        phase="validation",
        reset_on="validation_start",
        finalize_on="validation_end",
    ):
        self.num_classes = int(num_classes)
        self.average_object_size = torch.as_tensor(average_object_size).float()
        self.prediction = prediction
        self.target = target
        self.threshold = threshold
        self.fp_fraction = false_positive_fraction
        self.fn_fraction = false_negative_fraction
        self.prefix = prefix
        self.phase = phase
        self.reset_on = reset_on
        self.finalize_on = finalize_on
        self.reset()

    def reset(self):
        """Обнуляет накопленные object-level TP/FP/FN."""
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)

    def __call__(self):
        logits = self.context[self.prediction].detach()
        target = self.context[self.target].detach().bool()
        prediction = torch.sigmoid(logits) >= self.threshold
        dims = tuple(range(2, target.ndim))

        intersection = (prediction & target).sum(dims).float()
        target_area = target.sum(dims).float()
        pred_area = prediction.sum(dims).float()
        true_present = target_area > 0

        covered = intersection / target_area.clamp_min(1)
        average_size = self.average_object_size.to(logits.device).view(1, -1)
        false_positive = (~true_present) & (pred_area >= self.fp_fraction * average_size)
        detected = (true_present & (covered > 1 - self.fn_fraction)) | false_positive

        self.tp += (true_present & detected).sum(0).cpu()
        self.fp += (~true_present & detected).sum(0).cpu()
        self.fn += (true_present & ~detected).sum(0).cpu()

    def reaction(self, signal, source=None, **payload):
        if signal == self.finalize_on:
            self._publish()
        elif signal == self.reset_on:
            self.reset()

    def _publish(self):
        """Публикует метрики текущего окна и сообщает об их готовности."""
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        self.context[f"{self.prefix}.precision"] = precision
        self.context[f"{self.prefix}.recall"] = recall
        self.context[f"{self.prefix}.f1"] = f1
        self.reset()
        self.signal("metrics_ready", phase=self.phase, prefix=self.prefix)
