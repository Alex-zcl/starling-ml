"""Программная inference-интеграция nnU-Net v2."""

import torch

from ..core.module import Module


def _predictor_class():
    """Импортирует официальный nnUNetPredictor только при использовании."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise ImportError("nnU-Net integration requires nnunetv2") from exc
    return nnUNetPredictor


class NNUNetPredictorLoader(Module):
    """Инициализирует predictor один раз и хранит его между запросами."""

    def setup(
        self,
        trained_model_folder,
        output="nnunet.predictor",
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
        device="cuda",
        **predictor_kwargs,
    ):
        predictor = _predictor_class()(
            device=torch.device(device),
            **predictor_kwargs,
        )
        predictor.initialize_from_trained_model_folder(
            trained_model_folder,
            use_folds=tuple(use_folds),
            checkpoint_name=checkpoint_name,
        )
        self.context[output] = predictor


class NNUNetPredictArray(Module):
    """Предсказывает одну numpy-array с уже известными nnU-Net properties."""

    def setup(
        self,
        predictor,
        image,
        properties,
        output="nnunet.segmentation",
        save_probabilities=False,
    ):
        self.predictor = predictor
        self.image = image
        self.properties = properties
        self.output = output
        self.save_probabilities = save_probabilities

    def __call__(self):
        self.context[self.output] = self.predictor.predict_single_npy_array(
            self.context[self.image],
            self.context[self.properties],
            None,
            None,
            self.save_probabilities,
        )
