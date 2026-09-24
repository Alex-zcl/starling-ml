"""Inference и training-примитивы Hugging Face Diffusers."""

import torch

from ..core.module import Module


def _diffusers():
    """Лениво импортирует diffusers, не утяжеляя базовую установку."""
    try:
        import diffusers
    except ImportError as exc:
        raise ImportError("Diffusers integration requires diffusers") from exc
    return diffusers


def _select(result, selector):
    """Извлекает sample/attribute или элемент tuple."""
    if isinstance(selector, int):
        return result[selector]
    if isinstance(result, dict):
        return result[selector]
    return getattr(result, selector)


class DiffusionPipelineLoader(Module):
    """Загружает полный DiffusionPipeline для inference-сценариев."""

    def setup(self, pretrained, output="diffusion.pipeline", device=None, **kwargs):
        pipeline = _diffusers().DiffusionPipeline.from_pretrained(pretrained, **kwargs)
        if device is not None:
            pipeline = pipeline.to(device)
        self.context[output] = pipeline


class DiffusionPipelineCall(Module):
    """Вызывает готовый inference pipeline с Context-аргументами."""

    def setup(self, pipeline, inputs, output="diffusion.output", **kwargs):
        self.pipeline = pipeline
        self.inputs = dict(inputs)
        self.output = output
        self.kwargs = kwargs

    def __call__(self):
        values = {name: self.context[key] for name, key in self.inputs.items()}
        self.context[self.output] = self.pipeline(**values, **self.kwargs)


class DiffusionSchedulerLoader(Module):
    """Загружает конкретный scheduler class для train или замены inference schedule."""

    def setup(
        self,
        pretrained,
        scheduler_class="DDPMScheduler",
        output="diffusion.scheduler",
        **kwargs,
    ):
        scheduler_type = getattr(_diffusers(), scheduler_class)
        self.context[output] = scheduler_type.from_pretrained(pretrained, **kwargs)


class DiffusionNoiseSampler(Module):
    """Создаёт Gaussian noise и случайные timesteps для текущего clean sample."""

    def setup(
        self,
        sample,
        scheduler,
        noise_output="diffusion.noise",
        timestep_output="diffusion.timesteps",
    ):
        self.sample = sample
        self.scheduler = scheduler
        self.noise_output = noise_output
        self.timestep_output = timestep_output

    def __call__(self):
        sample = self.context[self.sample]
        scheduler = self.context[self.scheduler]
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (sample.shape[0],),
            device=sample.device,
            dtype=torch.long,
        )
        self.context[self.noise_output] = torch.randn_like(sample)
        self.context[self.timestep_output] = timesteps


class DiffusionAddNoise(Module):
    """Применяет scheduler.add_noise как отдельную видимую pipeline-операцию."""

    def setup(
        self,
        scheduler,
        sample,
        noise,
        timesteps,
        output="diffusion.noisy_sample",
    ):
        self.scheduler = scheduler
        self.sample = sample
        self.noise = noise
        self.timesteps = timesteps
        self.output = output

    def __call__(self):
        scheduler = self.context[self.scheduler]
        self.context[self.output] = scheduler.add_noise(
            self.context[self.sample],
            self.context[self.noise],
            self.context[self.timesteps],
        )


class DiffusionModelForward(Module):
    """Вызывает UNet/Transformer diffusion model без полного Pipeline."""

    def setup(
        self,
        model,
        sample,
        timesteps,
        conditions=None,
        output="model.noise_prediction",
        selector="sample",
    ):
        self.model = model
        self.sample = sample
        self.timesteps = timesteps
        self.conditions = dict(conditions or {})
        self.output = output
        self.selector = selector

    def __call__(self):
        kwargs = {name: self.context[key] for name, key in self.conditions.items()}
        result = self.model(
            self.context[self.sample],
            self.context[self.timesteps],
            **kwargs,
        )
        self.context[self.output] = _select(result, self.selector)
