"""Чистые overlap objectives: pixel statistics и внешняя важность разделены."""
import torch
import torch.nn.functional as F
from .objectives import Objective, stable, positive_weight, divide
from .reduction import element_map, pair_objective, weighted_mean
from .tensors import channel_view, sample_view, case_class_view


def activate(logits, activation):
    logits = stable(logits)
    if activation in (None, "none"):
        return logits
    if activation == "sigmoid":
        return logits.sigmoid()
    if activation == "softmax":
        return logits.softmax(1)
    raise ValueError(f"Unknown activation: {activation}")


def prepare_target(target, prediction, mode, ignore_index=-100, valid_mask=None):
    """Ignore labels исключаются до one_hot; unknown не превращается в negative."""
    if mode in {"binary", "multilabel"}:
        if target.shape != prediction.shape:
            raise ValueError("Binary target and prediction shapes must match")
        valid = (target != ignore_index) & element_map(valid_mask, prediction).bool()
        target = torch.where(valid, target, torch.zeros_like(target)).to(prediction.dtype)
    elif mode == "multiclass":
        if target.ndim == prediction.ndim:
            if target.shape[1] != 1:
                raise ValueError("Overlap multiclass target must contain class indices")
            target = target.squeeze(1)
        valid = target != ignore_index
        if valid_mask is not None:
            valid = valid & element_map(valid_mask, target.float(), channel=False).bool()
        if target.is_floating_point() and not torch.equal(target[valid], target[valid].round()):
            raise ValueError("Multiclass overlap requires integer labels")
        safe = torch.where(valid, target, torch.zeros_like(target)).long()
        if ((safe < 0) | (safe >= prediction.shape[1])).any():
            raise ValueError("Target class outside class mapping")
        target = F.one_hot(safe, prediction.shape[1]).movedim(-1, 1).to(prediction.dtype)
        valid = valid.unsqueeze(1).expand_as(prediction)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    valid = valid.to(prediction.dtype) * element_map(valid_mask, prediction).bool()
    if ((target < 0) | (target > 1)).any():
        raise ValueError("Target probabilities must lie in [0,1]")
    return target, valid


def prepare_overlap_target(target, prediction, mode):
    return prepare_target(target, prediction, mode)[0]


def _ratio(tp, fp, fn, kind, smooth, alpha, beta):
    if kind == "dice":
        numerator, denominator = 2 * tp, 2 * tp + fp + fn
    elif kind == "iou":
        numerator, denominator = tp, tp + fp + fn
    elif kind == "tversky":
        numerator, denominator = tp, tp + alpha * fp + beta * fn
    else:
        raise ValueError(f"Unknown overlap kind: {kind}")
    # Для пустого prediction и target standard similarity равна 1 даже при smooth=0.
    total = denominator + smooth
    score = torch.where(total > 0, divide(numerator + smooth, total), torch.ones_like(total))
    return 1 - score


def overlap_from_probabilities(prediction, target, kind="dice", aggregation="sample",
        element_weight=None, case_class_weight=None, class_weight=None, sample_weight=None,
        empty_target="ignore", smooth=1., eps=1e-7, alpha=.5, beta=.5,
        valid_mask=None, reduction="pair_mean", positive_weight=None, negative_weight=None,
        empty_cost=1., include_classes=None, return_objective=False,
        present_weight=None, empty_weight=None):
    prediction, target = stable(prediction), target.to(stable(prediction).dtype)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("Overlap requires matching [B,C,...] tensors")
    if smooth < 0 or alpha < 0 or beta < 0:
        raise ValueError("Overlap smoothing and Tversky factors must be non-negative")
    if empty_target == "penalize":
        empty_target = "standard"
    if empty_target not in {"ignore", "standard", "false_positive"}:
        raise ValueError("Invalid empty_target")
    if aggregation not in {"sample", "batch", "batch_stats"}:
        raise ValueError("Invalid aggregation")
    mask = element_map(valid_mask, prediction).bool().to(prediction.dtype)
    prediction = torch.where(mask.bool(), prediction, torch.zeros_like(prediction))
    target = torch.where(mask.bool(), target, torch.zeros_like(target))
    weight = mask * element_map(element_weight, prediction)
    # Positive/negative maps относятся к target-domain, а не к готовому Dice.
    if positive_weight is not None or negative_weight is not None:
        pos = channel_view(1. if positive_weight is None else positive_weight, prediction)
        neg = channel_view(1. if negative_weight is None else negative_weight, prediction)
        positive_weight_check = positive_weight if positive_weight is not None else 1.
        from .objectives import positive_weight as check_weight
        check_weight(positive_weight_check, prediction)
        check_weight(1. if negative_weight is None else negative_weight, prediction)
        weight = weight * (target * pos + (1-target) * neg)
    if prediction.ndim == 2:
        prediction, target, weight, mask = [v.unsqueeze(-1) for v in (prediction, target, weight, mask)]
    dims = tuple(range(2, prediction.ndim))
    channels = prediction.shape[1]
    cw = torch.ones(channels, device=prediction.device, dtype=prediction.dtype)
    if class_weight is not None:
        cw = cw * positive_weight_fn(class_weight, prediction)
    if include_classes is not None:
        selected = torch.zeros_like(cw)
        selected[list(include_classes)] = 1
        cw = cw * selected
    cost = positive_weight_fn(empty_cost, prediction).expand(channels)
    present_factor = positive_weight_fn(1. if present_weight is None else present_weight, prediction).expand(channels)
    empty_factor = positive_weight_fn(1. if empty_weight is None else empty_weight, prediction).expand(channels)

    if aggregation in {"batch", "batch_stats"}:
        if sample_weight is not None:
            weight = weight * sample_view(sample_weight, prediction)
            mask = mask * sample_view(sample_weight, prediction)
        if case_class_weight is not None:
            weight = weight * case_class_view(case_class_weight, prediction)
            mask = mask * case_class_view(case_class_weight, prediction)
        dims = (0, *dims)
    stats = ((weight*prediction*target).sum(dims),
             (weight*prediction*(1-target)).sum(dims),
             (weight*(1-prediction)*target).sum(dims),
             (mask*target).sum(dims), weight.sum(dims))

    def values(tp, fp, fn, present, support):
        loss = _ratio(tp, fp, fn, kind, smooth, alpha, beta)
        empty = present == 0
        if empty_target == "false_positive":
            loss = torch.where(empty, divide(tp+fp, support), loss)
        loss = torch.where(empty, loss * cost, loss)
        valid = (support > 0) & ((~empty) if empty_target == "ignore" else torch.ones_like(empty))
        return loss, valid

    signature = (kind, aggregation, empty_target, smooth, alpha, beta,
                 tuple(cw.detach().cpu().tolist()), tuple(cost.detach().cpu().tolist()),
                 tuple(present_factor.detach().cpu().tolist()), tuple(empty_factor.detach().cpu().tolist()))
    if aggregation == "sample":
        loss, valid = values(*stats)
        cases = torch.where(stats[3] > 0, present_factor, empty_factor)
        if case_class_weight is not None:
            cases = cases * positive_weight_fn(case_class_weight, prediction)
        obj = pair_objective(loss, valid, cw, cases, sample_weight, reduction)
        obj.signature = (signature, obj.signature)
    else:
        def finish(*parts):
            loss, valid = values(*parts)
            cases = torch.where(parts[3] > 0, present_factor, empty_factor)
            return weighted_mean(loss, cw * valid * cases, dim=0)
        obj = Objective(stats, finish, signature)
    return obj if return_objective else obj.tensor()


# Имя не пересекается с аргументом positive_weight публичного API.
positive_weight_fn = positive_weight


def overlap_objective(logits, target, activation="sigmoid", mode="multilabel",
                      ignore_index=-100, valid_mask=None, **kwargs):
    prediction = activate(logits, activation)
    target, mask = prepare_target(target, prediction, mode, ignore_index, valid_mask)
    return overlap_from_probabilities(prediction, target, valid_mask=mask, **kwargs)


def generalized_dice_objective(logits, target, activation="softmax", mode="multiclass",
        element_weight=None, class_weight=None, sample_weight=None, case_class_weight=None,
        empty_target="ignore", smooth=1., eps=1e-7, valid_mask=None, ignore_index=-100,
        include_classes=None, return_objective=False):
    prediction = activate(logits, activation)
    target, mask = prepare_target(target, prediction, mode, ignore_index, valid_mask)
    weight = mask * element_map(element_weight, prediction)
    if sample_weight is not None:
        weight = weight * sample_view(sample_weight, prediction)
    if case_class_weight is not None:
        weight = weight * case_class_view(case_class_weight, prediction)
    dims = (0, *range(2, prediction.ndim))
    tp = (weight*prediction*target).sum(dims)
    mass = (weight*(prediction+target)).sum(dims)
    present = (weight*target).sum(dims)
    support = weight.sum(dims)
    cw = torch.ones_like(tp) if class_weight is None else positive_weight(class_weight, tp).expand_as(tp)
    if include_classes is not None:
        selected = torch.zeros_like(cw); selected[list(include_classes)] = 1; cw = cw * selected
    if empty_target not in {"ignore", "standard", "penalize"}:
        raise ValueError("Generalized Dice supports ignore or standard")
    if smooth < 0:
        raise ValueError("smooth must be non-negative")
    def finish(tp, mass, present, support):
        active = (support > 0) & ((present > 0) if empty_target == "ignore" else torch.ones_like(present).bool())
        factors = cw * active
        denominator = (factors*mass).sum() + smooth
        numerator = 2*(factors*tp).sum() + smooth
        value = 1 - torch.where(denominator > 0, divide(numerator, denominator), torch.ones_like(denominator))
        return value * (factors.sum() > 0)
    obj = Objective((tp,mass,present,support), finish, ("generalized",empty_target,smooth,tuple(cw.detach().cpu().tolist())))
    return obj if return_objective else obj.tensor()


def overlap_loss(logits, target, kind="dice", activation="sigmoid", mode="multilabel",
                 pixel_weight=None, smooth=1., eps=1e-7):
    """Compatibility per-case API; новая реализация поддерживает classification shape."""
    prediction = activate(logits, activation)
    target, mask = prepare_target(target, prediction, mode)
    weight = mask * element_map(pixel_weight, prediction)
    if prediction.ndim == 2:
        prediction,target,weight = [v.unsqueeze(-1) for v in (prediction,target,weight)]
    dims = tuple(range(2,prediction.ndim))
    tp=(weight*prediction*target).sum(dims);fp=(weight*prediction*(1-target)).sum(dims);fn=(weight*(1-prediction)*target).sum(dims)
    return _ratio(tp,fp,fn,kind,smooth,.5,.5), (weight*target).sum(dims)


def reduce_overlap(loss,target_sum,class_weight=None,empty_target="ignore"):
    valid = target_sum > 0 if empty_target == "ignore" else torch.ones_like(target_sum).bool()
    return pair_objective(loss,valid,class_weight=class_weight).tensor()
