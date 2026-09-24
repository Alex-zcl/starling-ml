"""Чистые функции накопления confusion-статистик и метрик."""

import torch


def multilabel_stats(logits, target, threshold=0.5, valid_mask=None, ignore_index=-100):
    """Считает TP/FP/FN/TN по классам для независимых sigmoid-каналов."""
    prediction = torch.sigmoid(logits) >= threshold
    from .reduction import element_map
    valid = (target != ignore_index) & element_map(valid_mask, logits).bool()
    if target.ndim == 1:
        prediction, target, valid = prediction[:, None], target[:, None], valid[:, None]
    target = target.bool()
    dims = tuple(i for i in range(target.ndim) if i != 1)

    tp = ((prediction & target) & valid).sum(dims).float()
    fp = ((prediction & ~target) & valid).sum(dims).float()
    fn = ((~prediction & target) & valid).sum(dims).float()
    tn = ((~prediction & ~target) & valid).sum(dims).float()
    return tp, fp, fn, tn


def multiclass_stats(logits, target, num_classes, valid_mask=None, ignore_index=-100):
    """Строит confusion matrix и переводит её в one-vs-rest статистики."""
    prediction = logits.argmax(dim=1).reshape(-1)
    if target.ndim == logits.ndim:
        target = target.squeeze(1)
    from .reduction import element_map
    valid = (target != ignore_index) & element_map(valid_mask, target.float(), channel=False).bool()
    target = target.reshape(-1)[valid.reshape(-1)].long()
    prediction = prediction[valid.reshape(-1)]
    if ((target < 0) | (target >= num_classes)).any():
        raise ValueError("Metric target outside class mapping")

    confusion = torch.bincount(
        target * num_classes + prediction,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).float()
    tp = confusion.diag()
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    tn = confusion.sum() - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_stats(tp, fp, fn, tn, eps=1e-7):
    """Вычисляет интерпретируемые метрики из накопленных статистик."""
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
    }
