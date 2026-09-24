"""Простые preprocessing и postprocessing операции над тензорами."""

import torch
import torch.nn.functional as F

from ...core.module import Module


class Sigmoid(Module):
    """Применяет sigmoid как отдельный inference/postprocessing шаг."""

    def setup(self, input, output):
        self.input, self.output = input, output

    def __call__(self):
        self.context[self.output] = torch.sigmoid(self.context[self.input])


class Softmax(Module):
    """Применяет softmax по указанному измерению."""

    def setup(self, input, output, dim=1):
        self.input, self.output, self.dim = input, output, dim

    def __call__(self):
        self.context[self.output] = torch.softmax(self.context[self.input], dim=self.dim)


class Threshold(Module):
    """Преобразует вероятности в boolean mask по порогу."""

    def setup(self, input, output, threshold=0.5):
        self.input, self.output, self.threshold = input, output, threshold

    def __call__(self):
        self.context[self.output] = self.context[self.input] >= self.threshold


class Argmax(Module):
    """Выбирает индекс наиболее вероятного класса."""

    def setup(self, input, output, dim=1):
        self.input, self.output, self.dim = input, output, dim

    def __call__(self):
        self.context[self.output] = self.context[self.input].argmax(self.dim)


class ResizeSegmentation(Module):
    """Изменяет image и mask согласованно, но разными interpolation modes."""

    def setup(self, image, mask, size, output_image=None, output_mask=None):
        self.image = image
        self.mask = mask
        self.size = tuple(size)
        self.output_image = output_image or image
        self.output_mask = output_mask or mask

    def __call__(self):
        image = F.interpolate(
            self.context[self.image],
            self.size,
            mode="bilinear",
            align_corners=False,
        )
        mask = F.interpolate(self.context[self.mask].float(), self.size, mode="nearest")
        self.context[self.output_image] = image
        self.context[self.output_mask] = mask


class GaussianNoise(Module):
    """Добавляет Gaussian noise без скрытого изменения исходного ключа."""

    def setup(self, input, output=None, std=0.005, clamp=(0.0, 1.0)):
        self.input = input
        self.output = output or input
        self.std = std
        self.clamp = clamp

    def __call__(self):
        value = self.context[self.input]
        value = value + torch.randn_like(value) * self.std
        if self.clamp is not None:
            value = value.clamp(*self.clamp)
        self.context[self.output] = value


class RandomFlip(Module):
    """Применяет одно случайное отражение одновременно к image и mask."""

    def setup(
        self,
        image,
        mask,
        probability=0.5,
        dims=(-1,),
        output_image=None,
        output_mask=None,
    ):
        self.image = image
        self.mask = mask
        self.probability = probability
        self.dims = tuple(dims)
        self.output_image = output_image or image
        self.output_mask = output_mask or mask

    def __call__(self):
        image = self.context[self.image]
        mask = self.context[self.mask]
        if torch.rand(()) < self.probability:
            image = torch.flip(image, self.dims)
            mask = torch.flip(mask, self.dims)
        self.context[self.output_image] = image
        self.context[self.output_mask] = mask
