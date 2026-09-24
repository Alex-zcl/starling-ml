"""Небольшие synthetic data modules для smoke-тестов и примеров."""

import torch

from ...core.module import Module


class SyntheticSegmentationSpec(Module):
    """Публикует metadata отдельно от генерации batch.

    Model может зависеть от числа каналов и классов, не создавая train/validation
    batch во время setup. Это делает setup-граф явным и не смешивает lifecycle.
    """

    def setup(
        self,
        classes=2,
        in_channels=1,
        num_classes_key="data.num_classes",
        in_channels_key="data.in_channels",
    ):
        self.context[num_classes_key] = int(classes)
        self.context[in_channels_key] = int(in_channels)


class SyntheticSegmentationBatch(Module):
    """Генерирует один простой multilabel segmentation batch."""

    def setup(
        self,
        batch_size=8,
        image_size=32,
        classes=2,
        seed=42,
        output_image="batch.image",
        output_mask="batch.mask",
        object_probability=0.75,
    ):
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.classes = int(classes)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.output_image = output_image
        self.output_mask = output_mask
        self.object_probability = float(object_probability)

    def __call__(self):
        size = self.image_size
        masks = torch.zeros(self.batch_size, self.classes, size, size)

        for batch_index in range(self.batch_size):
            for class_index in range(self.classes):
                if torch.rand((), generator=self.generator) < self.object_probability:
                    height = int(torch.randint(4, 10, (), generator=self.generator))
                    width = int(torch.randint(4, 10, (), generator=self.generator))
                    y = int(torch.randint(0, size - height, (), generator=self.generator))
                    x = int(torch.randint(0, size - width, (), generator=self.generator))
                    masks[
                        batch_index,
                        class_index,
                        y : y + height,
                        x : x + width,
                    ] = 1

        image = masks.sum(1, keepdim=True).clamp(max=1)
        noise = 0.15 * torch.randn(image.shape, generator=self.generator)
        image = (image + noise).clamp(0, 1)
        self.context[self.output_image] = image
        self.context[self.output_mask] = masks


class SyntheticSegmentationData(SyntheticSegmentationBatch):
    """Compatibility-вариант, который также публикует metadata в setup."""

    def setup(self, batch_size=8, image_size=32, classes=2, seed=42):
        super().setup(batch_size, image_size, classes, seed)
        self.context["data.num_classes"] = int(classes)
        self.context["data.in_channels"] = 1
