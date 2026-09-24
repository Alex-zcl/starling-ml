"""Независимая диагностика gradients."""

import torch

from ...core.module import Module


class GradientMonitor(Module):
    """Считает общий gradient norm и при необходимости проверяет NaN/Inf."""

    def setup(self, model, output="monitor.grad_norm", fail_on_nonfinite=True):
        self.model = model
        self.output = output
        self.fail_on_nonfinite = fail_on_nonfinite

    def __call__(self):
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        value = torch.stack(norms).norm(2) if norms else torch.tensor(0.0)
        if self.fail_on_nonfinite and not torch.isfinite(value):
            raise FloatingPointError("Non-finite gradient norm")
        self.context[self.output] = float(value.detach().cpu())


# Старое имя оставлено для совместимости с ранней версией проекта.
GradientCheck = GradientMonitor
