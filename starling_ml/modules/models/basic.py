"""Небольшие встроенные PyTorch-модели для тестов и примеров."""

import torch
from torch import nn

from ...core.module import Module


class TinySegmenter(Module):
    """Создаёт компактную 2D segmentation network для smoke-test."""

    def setup(
        self,
        in_channels,
        classes,
        hidden=12,
        seed=42,
        output="model.instance",
    ):
        torch.manual_seed(int(seed))
        model = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, classes, 1),
        )
        self.context[output] = model


# Старый import path остаётся рабочим, но новая реализация живёт в integrations.
from ...integrations.smp import SMPModel  # noqa: E402
