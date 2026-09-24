"""Чистые функции создания и стабилизации весов."""

import torch


def normalize_mean(weights, eps=1e-7):
    """Нормирует средний вес к единице, сохраняя масштаб loss."""
    weights = torch.as_tensor(weights).float()
    return weights / weights.mean().clamp_min(eps)


def inverse_frequency_weights(
    frequency,
    gamma=1.0,
    eps=1e-6,
    normalize=True,
    min_value=None,
    max_value=None,
):
    """Преобразует частоты в устойчивые inverse-frequency weights.

    gamma позволяет плавно выбирать силу компенсации дисбаланса: 0 означает
    одинаковые веса, 0.5 — inverse square root, 1 — полный inverse frequency.
    """
    frequency = torch.as_tensor(frequency).float()
    weights = (frequency + eps).pow(-float(gamma))
    if normalize:
        weights = normalize_mean(weights, eps)
    if min_value is not None or max_value is not None:
        weights = weights.clamp(min=min_value, max=max_value)
    return weights


def adaptive_binary_weights(precision, recall, b=1.0, balance_band=0.1, eps=1e-7):
    """Строит positive/negative factors из дисбаланса precision и recall."""
    precision = torch.as_tensor(precision).float()
    recall = torch.as_tensor(recall).float()
    f1 = 2 * precision * recall / (precision + recall + eps)

    valid = (precision > 0) & (recall > 0)
    positive = torch.where(valid, b * precision / (f1 + eps), torch.full_like(f1, 2.0))
    negative = torch.where(valid, recall / (b * (f1 + eps)), torch.full_like(f1, 0.5))

    ratio = recall / (precision + eps)
    balanced = valid & (ratio > 1 - balance_band) & (ratio < 1 + balance_band)
    positive = torch.where(balanced, torch.ones_like(positive), positive)
    negative = torch.where(balanced, torch.ones_like(negative), negative)
    return positive, negative, positive.clone()


def presence_case_weights(target, present_weight, empty_weight, mode="multilabel", num_classes=None,
                          valid_mask=None, ignore_index=-100):
    """Unknown не считается empty; полностью невалидная пара получает нулевой вес."""
    from .losses import prepare_target
    from .objectives import positive_weight
    target = torch.as_tensor(target)
    if mode == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes is required")
        shape = (target.shape[0], num_classes, *target.shape[1:])
        reference = torch.zeros(shape, device=target.device)
    else:
        if target.ndim == 1:
            target = target[:, None]
        reference = torch.zeros_like(target, dtype=torch.float32)
    values, valid = prepare_target(target, reference, mode, ignore_index, valid_mask)
    present = (values * valid).reshape(values.shape[0], values.shape[1], -1).sum(-1) > 0
    known = valid.reshape(values.shape[0], values.shape[1], -1).sum(-1) > 0
    pos = positive_weight(present_weight, reference).reshape(1, -1)
    neg = positive_weight(empty_weight, reference).reshape(1, -1)
    return torch.where(present, pos, neg) * known


def smooth_update(old, new, momentum=0.0, max_change=None):
    """Стабилизирует динамические веса через EMA и ограничение шага."""
    old = torch.as_tensor(old).float()
    new = torch.as_tensor(new, device=old.device).float()
    value = float(momentum) * old + (1 - float(momentum)) * new
    if max_change is not None:
        delta = (value - old).clamp(min=-max_change, max=max_change)
        value = old + delta
    return value
