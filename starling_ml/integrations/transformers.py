"""Тонкие адаптеры Hugging Face Transformers."""

from contextlib import nullcontext

import torch

from ..core.module import Module


def _transformers():
    """Импортирует optional dependency только при использовании интеграции."""
    try:
        import transformers
    except ImportError as exc:
        raise ImportError("Transformers integration requires transformers") from exc
    return transformers


def _select(result, selector):
    """Извлекает значение из ModelOutput, dict или tuple."""
    if isinstance(selector, int):
        return result[selector]
    if isinstance(result, dict):
        return result[selector]
    return getattr(result, selector)


class HFAutoProcessor(Module):
    """Загружает AutoProcessor для multimodal и Florence-like моделей."""

    def setup(self, pretrained, output="hf.processor", **kwargs):
        processor = _transformers().AutoProcessor.from_pretrained(pretrained, **kwargs)
        self.context[output] = processor


class HFAutoTokenizer(Module):
    """Загружает AutoTokenizer без собственного registry архитектур."""

    def setup(self, pretrained, output="hf.tokenizer", **kwargs):
        tokenizer = _transformers().AutoTokenizer.from_pretrained(pretrained, **kwargs)
        self.context[output] = tokenizer


class HFAutoModel(Module):
    """Загружает выбранный AutoModel* класс по имени."""

    def setup(
        self,
        pretrained,
        auto_class="AutoModel",
        output="model.instance",
        training=True,
        **kwargs,
    ):
        transformers = _transformers()
        model_class = getattr(transformers, auto_class)
        model = model_class.from_pretrained(pretrained, **kwargs)
        model.train(bool(training))
        self.context[output] = model


class HFProcess(Module):
    """Превращает Context-входы в batch, ожидаемый processor/tokenizer."""

    def setup(
        self,
        processor,
        inputs,
        output="batch.hf",
        return_tensors="pt",
        **kwargs,
    ):
        self.processor = processor
        self.inputs = dict(inputs)
        self.output = output
        self.return_tensors = return_tensors
        self.kwargs = kwargs

    def __call__(self):
        values = {name: self.context[key] for name, key in self.inputs.items()}
        self.context[self.output] = self.processor(
            **values,
            return_tensors=self.return_tensors,
            **self.kwargs,
        )


class HFForward(Module):
    """Вызывает PreTrainedModel и раскладывает ModelOutput по Context keys."""

    def setup(
        self,
        model,
        inputs="batch.hf",
        output="model.hf_output",
        outputs=None,
        training=True,
        grad=None,
        autocast=False,
        device_type="cuda",
    ):
        self.model = model
        self.inputs = inputs
        self.output = output
        self.outputs = outputs
        self.training = bool(training)
        self.grad = self.training if grad is None else bool(grad)
        self.autocast = bool(autocast)
        self.device_type = device_type

    def __call__(self):
        self.model.train(self.training)
        batch = self.context[self.inputs]
        autocast = (
            torch.autocast(self.device_type, enabled=True)
            if self.autocast
            else nullcontext()
        )
        with torch.set_grad_enabled(self.grad), autocast:
            result = self.model(**batch)

        self.context[self.output] = result
        for context_key, selector in (self.outputs or {}).items():
            self.context[context_key] = _select(result, selector)


class HFGenerate(Module):
    """Отделяет autoregressive/multimodal generate от train forward."""

    def setup(self, model, inputs="batch.hf", output="model.generated", **kwargs):
        self.model = model
        self.inputs = inputs
        self.output = output
        self.kwargs = kwargs

    def __call__(self):
        self.model.eval()
        with torch.no_grad():
            self.context[self.output] = self.model.generate(
                **self.context[self.inputs], **self.kwargs
            )
