"""Достаточные статистики objective для точной accumulation и DDP."""
from dataclasses import dataclass
import torch


def stable(value):
    """Low-precision суммы выполняются в FP32, double остаётся double."""
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def divide(numerator, denominator):
    """Не вычисляет 0/0 даже в невыбранной ветке autograd."""
    valid = denominator > 0
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    return numerator / safe * valid.to(numerator.dtype)


def positive_weight(value, reference):
    weight = torch.as_tensor(value, device=reference.device, dtype=stable(reference).dtype)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Importance/cost must be finite and non-negative")
    return weight


def global_sum(value):
    """Differentiable all-reduce: DDP затем усредняет parameter gradients.

    Каждая rank вычисляет один и тот же global objective. Backward all-reduce
    суммирует его одинаковые производные, что компенсирует DDP averaging.
    """
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return value
    if value.requires_grad:
        from torch.distributed.nn.functional import all_reduce
        return all_reduce(value, op=dist.ReduceOp.SUM)
    result = value.clone()
    dist.all_reduce(result)
    return result


@dataclass
class Objective:
    """Хранит суммируемые stats и чистую функцию их финальной редукции."""
    stats: tuple
    finish: object
    signature: object

    def value(self, distributed=False):
        stats = tuple(global_sum(s) for s in self.stats) if distributed else self.stats
        return self.finish(*stats)

    def merge(self, other):
        if self.signature != other.signature or len(self.stats) != len(other.stats):
            raise ValueError("Objective or weights changed inside accumulation window")
        return Objective(tuple(a + b for a, b in zip(self.stats, other.stats)), self.finish, self.signature)

    def tensor(self):
        value = self.value()
        # Metadata сохраняет граф для exact accumulation, но не попадает в checkpoint.
        value._starling_ml_objective = self
        return value


@dataclass
class MixedObjective:
    terms: tuple
    mode: str = "sum"

    def value(self, distributed=False):
        values = [weight * obj.value(distributed) for weight, obj in self.terms]
        total = sum(values)
        if self.mode == "normalized_sum":
            denominator = sum(weight.abs() for weight, _ in self.terms)
            total = divide(total, denominator)
        return total

    def merge(self, other):
        if not isinstance(other, MixedObjective) or self.mode != other.mode or len(self.terms) != len(other.terms):
            raise ValueError("LossMixer changed inside accumulation window")
        terms = []
        for (wa, a), (wb, b) in zip(self.terms, other.terms):
            if not torch.equal(wa, wb):
                raise ValueError("Term weight changed inside accumulation window")
            terms.append((wa, a.merge(b)))
        return MixedObjective(tuple(terms), self.mode)

    def tensor(self):
        value = self.value()
        value._starling_ml_objective = self
        return value


def mean_objective(numerator, denominator, signature="mean"):
    return Objective((numerator, denominator), divide, signature)
