"""Минимальные модули для проверки универсального pipeline DSL."""

from starling_ml.core.module import Module


class FiniteSource(Module):
    """Пишет числа 0..limit-1, затем использует стандартный StopIteration."""

    def setup(self, limit, output="item"):
        self.limit = int(limit)
        self.output = output
        self.index = 0

    def __call__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.context[self.output] = self.index
        self.index += 1


class SumSink(Module):
    """Суммирует значения источника, чтобы проверить body блока iterate."""

    def setup(self, input="item", output="total"):
        self.input = input
        self.output = output
        self.context[output] = 0

    def __call__(self):
        self.context[self.output] = self.context[self.output] + self.context[self.input]
