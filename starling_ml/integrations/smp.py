"""Тонкая интеграция segmentation_models_pytorch."""

from ..core.module import Module


class SMPModel(Module):
    """Создаёт SMP-модель через официальный create_model factory."""

    def setup(
        self,
        architecture,
        encoder_name,
        in_channels,
        classes,
        encoder_weights=None,
        output="model.instance",
        **kwargs,
    ):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError("SMPModel requires segmentation-models-pytorch") from exc

        model = smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )
        self.context[output] = model
