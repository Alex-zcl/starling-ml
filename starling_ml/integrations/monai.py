"""Тонкие adapters для MONAI networks, transforms и inferers."""

from ..core.module import Module


def _monai():
    """Лениво импортирует MONAI."""
    try:
        import monai
    except ImportError as exc:
        raise ImportError("MONAI integration requires monai") from exc
    return monai


class MonaiNetwork(Module):
    """Создаёт сеть из monai.networks.nets по имени класса."""

    def setup(self, network_class, output="model.instance", **kwargs):
        network_type = getattr(_monai().networks.nets, network_class)
        self.context[output] = network_type(**kwargs)


class MonaiCompose(Module):
    """Создаёт Compose из компактного списка transform-конфигураций."""

    def setup(self, transforms, input, output=None):
        monai = _monai()
        operations = []
        for item in transforms:
            transform_type = getattr(monai.transforms, item["class"])
            operations.append(transform_type(**item.get("params", {})))
        self.transform = monai.transforms.Compose(operations)
        self.input = input
        self.output = output or input

    def __call__(self):
        self.context[self.output] = self.transform(self.context[self.input])


class MonaiSlidingWindowInferer(Module):
    """Выполняет sliding-window inference, оставляя predictor обычной PyTorch сетью."""

    def setup(
        self,
        predictor,
        input,
        roi_size,
        output="model.sliding_window_output",
        sw_batch_size=1,
        overlap=0.25,
        **kwargs,
    ):
        self.predictor = predictor
        self.input = input
        self.roi_size = roi_size
        self.output = output
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.kwargs = kwargs

    def __call__(self):
        infer = _monai().inferers.sliding_window_inference
        import torch
        was_training = self.predictor.training
        self.predictor.eval()
        try:
            with torch.no_grad():
                self.context[self.output] = infer(
                    inputs=self.context[self.input], roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size, predictor=self.predictor,
                    overlap=self.overlap, **self.kwargs,
                )
        finally:
            self.predictor.train(was_training)
