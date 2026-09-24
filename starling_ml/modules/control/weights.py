"""Модули, которые создают веса, но не применяют их к loss."""

import torch

from ...core.module import Module
from ...ops.weights import (
    adaptive_binary_weights,
    inverse_frequency_weights,
    normalize_mean,
    presence_case_weights,
    smooth_update,
)


class ConstantWeight(Module):
    """Создаёт scalar или tensor weight из конфигурации."""

    def setup(self, output, value):
        self.output = output
        self.context[output] = torch.as_tensor(value).float()


class FrequencyWeights(Module):
    """Строит inverse-frequency weights при старте или явном вызове.

    Частота может быть задана прямо в конфиге или лежать в Context. Runtime
    ключ передаётся строкой, чтобы его текущее значение читалось при пересчёте.
    """

    def setup(
        self,
        output,
        frequency=None,
        frequency_key=None,
        gamma=1.0,
        normalize=True,
        min_value=None,
        max_value=None,
        signal="run_started",
    ):
        if frequency is None and frequency_key is None:
            raise ValueError("FrequencyWeights needs frequency or frequency_key")
        self.output = output
        self.frequency = frequency
        self.frequency_key = frequency_key
        self.gamma = gamma
        self.normalize = normalize
        self.min_value = min_value
        self.max_value = max_value
        self.listen_signal = signal

        # Если значения заданы прямо, результат безопасно создать уже в setup.
        if frequency is not None:
            self._update()

    def __call__(self):
        self._update()

    def reaction(self, signal, source=None, **payload):
        if signal == self.listen_signal:
            self._update()
            self.signal("weights_changed")

    def _update(self):
        frequency = (
            self.context[self.frequency_key]
            if self.frequency_key is not None
            else self.frequency
        )
        self.context[self.output] = inverse_frequency_weights(
            frequency,
            gamma=self.gamma,
            normalize=self.normalize,
            min_value=self.min_value,
            max_value=self.max_value,
        )


class PresenceWeights(Module):
    """Создаёт [B, C] вес для present и empty масок текущего batch."""

    def setup(
        self,
        target,
        output,
        present_weight,
        empty_weight,
        mode="multilabel",
        num_classes=None,
        valid_mask=None,
        ignore_index=-100,
    ):
        self.target = target
        self.output = output
        self.present_weight = present_weight
        self.empty_weight = empty_weight
        self.mode = mode
        self.num_classes = num_classes
        self.valid_mask, self.ignore_index = valid_mask, ignore_index

    def __call__(self):
        present = (
            self.context[self.present_weight]
            if isinstance(self.present_weight, str)
            else self.present_weight
        )
        empty = (
            self.context[self.empty_weight]
            if isinstance(self.empty_weight, str)
            else self.empty_weight
        )
        self.context[self.output] = presence_case_weights(
            self.context[self.target],
            present_weight=present,
            empty_weight=empty,
            mode=self.mode,
            num_classes=self.num_classes,
            valid_mask=self.context[self.valid_mask] if self.valid_mask else None,
            ignore_index=self.ignore_index,
        )


class WeightProduct(Module):
    """Комбинирует независимые base/schedule/adaptive factors в один вес.

    Каждый источник остаётся владельцем своего ключа, а loss читает только
    effective output. Это предотвращает ситуацию, когда несколько managers
    незаметно перезаписывают одну переменную.
    """

    def setup(
        self,
        inputs,
        output,
        normalize=False,
        min_value=None,
        max_value=None,
        signals=("weights_changed",),
        initial=1.0,
    ):
        if not inputs:
            raise ValueError("WeightProduct requires inputs")
        self.inputs = list(inputs)
        self.output = output
        self.normalize = normalize
        self.min_value = min_value
        self.max_value = max_value
        self.signals = set(signals)
        self.context[output] = torch.as_tensor(initial).float()

    def __call__(self):
        self._update()

    def reaction(self, signal, source=None, **payload):
        if signal in self.signals:
            self._update()

    def _update(self):
        values = []
        for key in self.inputs:
            value = self.context.get(key)
            if value is None:
                return
            values.append(value)
        result = torch.as_tensor(values[0]).float().clone()
        for value in values[1:]:
            result = result * torch.as_tensor(value, device=result.device).float()
        if self.normalize and result.ndim > 0:
            result = normalize_mean(result)
        if self.min_value is not None or self.max_value is not None:
            result = result.clamp(min=self.min_value, max=self.max_value)
        previous = self.context[self.output]
        self.context[self.output] = result
        if not torch.equal(torch.as_tensor(previous), result):
            self.signal("weights_changed")

    def ready(self):
        self._update()

    def state_dict(self):
        return {"value":self.context[self.output]}

    def load_state_dict(self,state):
        self.context[self.output]=state["value"]


class MetricAdaptiveWeights(Module):
    """Обновляет binary factors только по готовым validation-метрикам."""

    def setup(
        self,
        num_classes,
        precision="metrics.validation.precision",
        recall="metrics.validation.recall",
        positive="weights.positive",
        negative="weights.negative",
        class_weight="weights.class",
        phase="validation",
        metric_prefix="metrics.validation",
        b=1.0,
        momentum=0.5,
        max_change=None,
        normalize_class=True,
        min_value=None,
        max_value=None,
    ):
        self.precision = precision
        self.recall = recall
        self.positive = positive
        self.negative = negative
        self.class_weight = class_weight
        self.phase = phase
        self.metric_prefix = metric_prefix
        self.b = b
        self.momentum = momentum
        self.max_change = max_change
        self.normalize_class = normalize_class
        self.min_value = min_value
        self.max_value = max_value

        ones = torch.ones(int(num_classes))
        self.context[positive] = ones.clone()
        self.context[negative] = ones.clone()
        self.context[class_weight] = ones.clone()

    def reaction(self, signal, source=None, **payload):
        if signal != "metrics_ready" or payload.get("phase") != self.phase or payload.get("prefix") != self.metric_prefix:
            return

        new_positive, new_negative, new_class = adaptive_binary_weights(
            self.context[self.precision], self.context[self.recall], b=self.b
        )
        if self.normalize_class:
            new_class = normalize_mean(new_class)

        for key, new in (
            (self.positive, new_positive),
            (self.negative, new_negative),
            (self.class_weight, new_class),
        ):
            old = self.context[key]
            value = smooth_update(
                old, new, momentum=self.momentum, max_change=self.max_change
            )
            if self.min_value is not None or self.max_value is not None:
                value = value.clamp(min=self.min_value, max=self.max_value)
            self.context[key] = value
        self.signal("weights_changed")

    def state_dict(self):
        return {key:self.context[key] for key in (self.positive,self.negative,self.class_weight)}

    def load_state_dict(self,state):
        for key,value in state.items():self.context[key]=value


class LinearWeightSchedule(Module):
    """Вычисляет scalar weight напрямую из global step.

    Внутренний счётчик не используется: после возобновления запуска или
    gradient accumulation единственным источником времени остаётся run.step.
    """

    def setup(
        self,
        output,
        start,
        target,
        steps,
        step_key="run.step",
        signal="train_step_end",
    ):
        self.output = output
        self.start = float(start)
        self.target = float(target)
        self.steps = max(int(steps), 1)
        self.step_key = step_key
        self.listen_signal = signal
        self.context[output] = self.start

    def reaction(self, signal, source=None, **payload):
        if signal != self.listen_signal:
            return
        step = min(int(self.context[self.step_key]), self.steps)
        alpha = step / self.steps
        self.context[self.output] = self.start + alpha * (self.target - self.start)
        self.signal("weights_changed")

    def state_dict(self):
        return {"value":self.context[self.output]}

    def load_state_dict(self,state):
        self.context[self.output]=state["value"]


class DatasetWeights(Module):
    """Строит одну из шести ролей из соответствующих counts, без скрытого reuse."""
    def setup(self,statistics,level="element",balance="class",output="weights.class",
              positive_output="weights.positive",negative_output="weights.negative",
              unit="study",gamma=.5,normalization="expectation",bounds=None,
              missing="error",positive_fraction=.5,target_fraction=None):
        from ...ops.statistics import frequency_factors,binary_frequency_factors
        if level not in {"element","case"} or balance not in {"class","binary"}:
            raise ValueError("DatasetWeights level=element/case, balance=class/binary")
        if unit not in {"study","sample"}:
            raise ValueError("Case unit must be study or sample")
        suffix="element" if level=="element" else unit
        positive=statistics[f"positive_{suffix}_count"]
        negative=statistics[f"negative_{suffix}_count"]
        opts=dict(gamma=gamma,normalization=normalization,bounds=bounds,missing=missing)
        self.keys=[output] if balance=="class" else [positive_output,negative_output]
        if balance=="class":
            self.context[output]=frequency_factors(positive,target_fraction=target_fraction,**opts)
        else:
            pos,neg=binary_frequency_factors(positive,negative,positive_fraction=positive_fraction,**opts)
            self.context[positive_output]=pos;self.context[negative_output]=neg

    def state_dict(self):
        return {key:self.context[key] for key in self.keys}

    def load_state_dict(self,state):
        for key,value in state.items():self.context[key]=value


class TargetClassWeights(Module):
    """Создаёт spatial importance map по ИСТИННОМУ target class до overlap/CE."""
    def setup(self,target,weights,output,ignore_index=-100):
        self.target,self.weights,self.output,self.ignore_index=target,weights,output,ignore_index

    def __call__(self):
        target=self.context[self.target]
        weights=self.context[self.weights] if isinstance(self.weights,str) else self.weights
        weights=torch.as_tensor(weights,device=target.device)
        valid=target!=self.ignore_index
        safe=torch.where(valid,target,torch.zeros_like(target)).long()
        self.context[self.output]=weights[safe]*valid
