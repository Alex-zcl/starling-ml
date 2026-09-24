"""Маски, costs и importance не смешиваются неявно в знаменателе."""
import torch
from .objectives import Objective, stable, positive_weight, divide, mean_objective


def weighted_mean(values, weight=None, dim=None, eps=1e-12):
    values = stable(values)
    if weight is None:
        weight = torch.ones_like(values)
    weight = torch.broadcast_to(positive_weight(weight, values), values.shape)
    # Где weight=0, NaN input не должен загрязнять результат или gradient.
    masked = torch.where(weight > 0, values, torch.zeros_like(values))
    return divide((masked * weight).sum(dim=dim), weight.sum(dim=dim))


def element_map(weight, reference, channel=True):
    if weight is None:
        return torch.ones_like(stable(reference))
    weight = positive_weight(weight, reference)
    if channel and reference.ndim > 2 and weight.ndim == reference.ndim - 1 and weight.shape[0] == reference.shape[0]:
        weight = weight.unsqueeze(1)
    return torch.broadcast_to(weight, reference.shape)


def pair_objective(loss, valid, class_weight=None, case_class_weight=None,
                   sample_weight=None, reduction="pair_mean"):
    """Объединяет [B,C] ошибки; class_macro не зависит от числа cases класса."""
    loss = stable(loss)
    cw = torch.ones(loss.shape[1], device=loss.device, dtype=loss.dtype)
    if class_weight is not None:
        cw = positive_weight(class_weight, loss).expand(loss.shape[1])
    case = valid.to(loss.dtype)
    if case_class_weight is not None:
        case = case * positive_weight(case_class_weight, loss)
    if sample_weight is not None:
        case = case * positive_weight(sample_weight, loss).reshape(-1, 1)
    safe_loss = torch.where(case > 0, loss, torch.zeros_like(loss))
    signature = (reduction, tuple(cw.detach().cpu().tolist()))
    if reduction == "class_macro":
        def finish(num, den):
            active = (den > 0).to(num.dtype) * cw
            return divide((divide(num, den) * active).sum(), active.sum())
        return Objective(((safe_loss * case).sum(0), case.sum(0)), finish, signature)
    weight = case * cw.reshape(1, -1)
    if reduction == "pair_mean":
        return mean_objective((safe_loss * weight).sum(), weight.sum(), signature)
    if reduction == "sample_mean":
        # Sample weight действует только после внутренней class-нормировки.
        class_case = valid.to(loss.dtype) * cw.reshape(1, -1)
        if case_class_weight is not None:
            class_case = class_case * positive_weight(case_class_weight, loss)
        den = class_case.sum(1)
        values = divide((safe_loss * class_case).sum(1), den)
        sw = (den > 0).to(loss.dtype)
        if sample_weight is not None:
            sw = sw * positive_weight(sample_weight, loss)
        return mean_objective((values * sw).sum(), sw.sum(), signature)
    raise ValueError("reduction must be pair_mean, sample_mean or class_macro")


def pointwise_objective(loss, element_weight=None, valid_mask=None,
                        case_class_weight=None, class_weight=None,
                        sample_weight=None, reduction="pair_mean",
                        element_normalization="weight_sum"):
    """Сначала element mean [B,C], затем выбранный внешний reducer."""
    loss = stable(loss)
    if loss.ndim == 1:
        loss = loss[:, None]
        if element_weight is not None and torch.as_tensor(element_weight).ndim == 1:
            element_weight = torch.as_tensor(element_weight)[:, None]
        if valid_mask is not None and torch.as_tensor(valid_mask).ndim == 1:
            valid_mask = torch.as_tensor(valid_mask)[:, None]
    if loss.ndim < 2:
        raise ValueError("Expected batch dimension")
    mask = element_map(valid_mask, loss).bool().to(loss.dtype)
    weight = mask * element_map(element_weight, loss)
    if element_normalization not in {"weight_sum", "valid_count"}:
        raise ValueError("element_normalization must be weight_sum or valid_count")
    den_weight = weight if element_normalization == "weight_sum" else mask
    safe = torch.where(weight > 0, loss, torch.zeros_like(loss))
    if reduction == "element_mean":
        from .tensors import channel_view, sample_view, case_class_view
        outer = torch.ones_like(loss)
        for value, reshape in ((class_weight, channel_view), (sample_weight, sample_view), (case_class_weight, case_class_view)):
            if value is not None:
                outer = outer * reshape(positive_weight(value, loss), loss)
        return mean_objective((safe * weight * outer).sum(), (den_weight * outer).sum(), (reduction, element_normalization))
    if loss.ndim == 2:
        # Один элемент на case: перенос importance в внешний reducer предотвращает сокращение.
        if element_normalization == "weight_sum":
            cases = weight
            if case_class_weight is not None:
                cases = cases * positive_weight(case_class_weight, loss)
            return pair_objective(safe, mask > 0, class_weight, cases, sample_weight, reduction)
        values = safe * weight
        return pair_objective(values, mask > 0, class_weight, case_class_weight, sample_weight, reduction)
    dims = tuple(range(2, loss.ndim))
    den = den_weight.sum(dims)
    values = divide((safe * weight).sum(dims), den)
    valid = den > 0
    return pair_objective(values, valid, class_weight, case_class_weight, sample_weight, reduction)


def reduce_pointwise(loss, element_weight=None, case_class_weight=None,
                     class_weight=None, sample_weight=None, eps=1e-12,
                     valid_mask=None, reduction="sample_mean", element_normalization="weight_sum"):
    # Legacy function оставляет sample_mean; новые модули явно используют pair_mean.
    return pointwise_objective(loss, element_weight, valid_mask, case_class_weight,
                               class_weight, sample_weight, reduction, element_normalization).tensor()


def categorical_objective(loss, valid, element_weight=None, sample_weight=None,
                          reduction="element_mean", element_normalization="weight_sum"):
    """CE/token loss без фиктивной class axis и без ignored samples в denominator."""
    loss = stable(loss)
    mask = element_map(valid, loss, channel=False).bool().to(loss.dtype)
    weight = mask * element_map(element_weight, loss, channel=False)
    den_weight = weight if element_normalization == "weight_sum" else mask
    if element_normalization not in {"weight_sum", "valid_count"}:
        raise ValueError("Invalid element_normalization")
    safe = torch.where(weight > 0, loss, torch.zeros_like(loss))
    dims = tuple(range(1, loss.ndim))
    num = (safe * weight).sum(dims) if dims else safe * weight
    den = den_weight.sum(dims) if dims else den_weight
    sw = torch.ones_like(den) if sample_weight is None else positive_weight(sample_weight, den)
    if reduction == "element_mean":
        return mean_objective((num * sw).sum(), (den * sw).sum(), (reduction, element_normalization))
    if reduction == "sample_mean":
        sw = sw * (den > 0)
        return mean_objective((divide(num, den) * sw).sum(), sw.sum(), (reduction, element_normalization))
    raise ValueError("Categorical reduction must be element_mean or sample_mean")


def reduce_per_sample(loss, element_weight=None, sample_weight=None, eps=1e-12):
    return categorical_objective(loss, torch.ones_like(loss), element_weight,
                                 sample_weight, "sample_mean").tensor()


def reduce_classes(loss, class_weight=None, valid=None, eps=1e-12):
    weight = torch.ones_like(loss) if class_weight is None else positive_weight(class_weight, loss)
    if valid is not None:
        weight = weight * valid
    return weighted_mean(loss, weight, dim=0)
