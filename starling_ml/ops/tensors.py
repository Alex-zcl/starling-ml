"""Небольшие tensor-утилиты без зависимости от движка."""

import torch


def as_tensor(value, reference):
    """Переносит вес на устройство и dtype вычисляемого тензора."""
    if value is None:
        return None
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def channel_view(values, tensor):
    """Преобразует [C] в форму, совместимую с [B, C, ...]."""
    values = as_tensor(values, tensor)
    if values is None or values.ndim == 0:
        return values
    if tensor.ndim == 1:
        if values.numel() != 1:
            raise ValueError("[B] output has one channel; use sample_weight for [B] weights")
        return values.reshape(())
    if values.numel() not in (1, tensor.shape[1]):
        raise ValueError("Channel weight must be scalar or contain C entries")
    return values.view(1, -1, *([1] * (tensor.ndim - 2)))


def sample_view(values, tensor):
    """Преобразует [B] в форму, совместимую с [B, C, ...]."""
    values = as_tensor(values, tensor)
    if values is None or values.ndim == 0:
        return values
    return values.view(-1, *([1] * (tensor.ndim - 1)))


def case_class_view(values, tensor):
    """Преобразует [B, C] в форму, совместимую с [B, C, ...]."""
    values = as_tensor(values, tensor)
    if values is None or values.ndim <= 1:
        return values
    return values.view(values.shape[0], values.shape[1], *([1] * (tensor.ndim - 2)))


def scalar(value):
    """Безопасно превращает scalar tensor или число в обычный float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected a scalar tensor")
        return float(value.detach().cpu().item())
    return float(value)
