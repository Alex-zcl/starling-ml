"""Универсальный PyTorch forward без task-specific логики."""

from contextlib import nullcontext

import torch

from ..core.module import Module


class Forward(Module):
    """Вызывает модель с positional или keyword inputs и сохраняет outputs.

    Train и validation обычно используют два экземпляра этого класса с разными
    ключами и режимом grad. Это яснее скрытого переключения внутри Engine.
    """

    def setup(
        self,
        model,
        input=None,
        inputs=None,
        output="model.output",
        outputs=None,
        training=True,
        grad=None,
        autocast=False,
        device_type="cuda",
        autocast_dtype=None,
    ):
        if input is not None and inputs is not None:
            raise ValueError("Forward accepts either input or inputs, not both")
        self.model = model
        self.input = input
        self.inputs = inputs
        self.output = output
        self.outputs = outputs
        self.training = bool(training)
        self.grad = self.training if grad is None else bool(grad)
        self.autocast = bool(autocast)
        self.device_type = device_type
        self.autocast_dtype = getattr(torch, autocast_dtype) if autocast_dtype else None

    def __call__(self):
        self.model.train(self.training)
        args, kwargs = self._inputs()

        autocast = (
            torch.autocast(
                device_type=self.device_type,
                dtype=self.autocast_dtype,
                enabled=True,
            )
            if self.autocast
            else nullcontext()
        )
        with torch.set_grad_enabled(self.grad), autocast:
            result = self.model(*args, **kwargs)

        if self.outputs:
            for context_key, selector in self.outputs.items():
                self.context[context_key] = self._select(result, selector)
        else:
            self.context[self.output] = result

    def _inputs(self):
        """Преобразует конфигурацию ключей в args/kwargs модели."""
        if self.input is not None:
            return [self.context[self.input]], {}
        if isinstance(self.inputs, dict):
            return [], {name: self.context[key] for name, key in self.inputs.items()}
        if isinstance(self.inputs, list):
            return [self.context[key] for key in self.inputs], {}
        return [], {}

    @staticmethod
    def _select(result, selector):
        """Извлекает tuple item, dict key или attribute из сложного output."""
        if isinstance(selector, int):
            return result[selector]
        if isinstance(result, dict):
            return result[selector]
        return getattr(result, selector)
