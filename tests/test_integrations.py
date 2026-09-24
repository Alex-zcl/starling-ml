"""Проверки lazy imports и минимальных границ внешних API."""

import sys
import types
import unittest
from unittest.mock import patch

from starling_ml.core.context import Context


class IntegrationImportTests(unittest.TestCase):
    def test_lazy_imports(self):
        from starling_ml.integrations.clearml import ClearMLLogger
        from starling_ml.integrations.diffusers import DiffusionPipelineLoader
        from starling_ml.integrations.monai import MonaiNetwork
        from starling_ml.integrations.nnunet import NNUNetPredictorLoader
        from starling_ml.integrations.smp import SMPModel
        from starling_ml.integrations.tensorboard import TensorBoardLogger
        from starling_ml.integrations.transformers import HFAutoModel

        classes = [
            ClearMLLogger,
            DiffusionPipelineLoader,
            HFAutoModel,
            MonaiNetwork,
            NNUNetPredictorLoader,
            SMPModel,
            TensorBoardLogger,
        ]
        self.assertTrue(all(isinstance(value, type) for value in classes))

    def test_disabled_loggers_need_no_optional_packages(self):
        from starling_ml.integrations.clearml import ClearMLLogger
        from starling_ml.integrations.tensorboard import TensorBoardLogger

        context = Context(
            {},
            {
                "TB": {},
                "ClearML": {},
            },
        )
        TensorBoardLogger(context.view("TB")).setup(log_dir="unused", enabled=False)
        ClearMLLogger(context.view("ClearML")).setup(
            project_name="unused", task_name="unused", enabled=False
        )

    def test_smp_wrapper_calls_create_model(self):
        from starling_ml.integrations.smp import SMPModel

        sentinel = object()
        fake = types.ModuleType("segmentation_models_pytorch")
        calls = {}

        def create_model(**kwargs):
            calls.update(kwargs)
            return sentinel

        fake.create_model = create_model
        context = Context({}, {"SMP": {"creates": ["model.instance"]}})
        with patch.dict(sys.modules, {"segmentation_models_pytorch": fake}):
            SMPModel(context.view("SMP")).setup(
                architecture="fpn",
                encoder_name="resnet18",
                in_channels=1,
                classes=2,
            )

        self.assertIs(context.data["model.instance"], sentinel)
        self.assertEqual(calls["arch"], "fpn")

    def test_transformers_wrapper_uses_selected_auto_class(self):
        from starling_ml.integrations.transformers import HFAutoModel

        fake = types.ModuleType("transformers")
        model = types.SimpleNamespace(train=lambda value: setattr(model, "training", value))

        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                model.name = name
                model.kwargs = kwargs
                return model

        fake.AutoModelForCausalLM = AutoModelForCausalLM
        context = Context({}, {"HF": {"creates": ["model.instance"]}})
        with patch.dict(sys.modules, {"transformers": fake}):
            HFAutoModel(context.view("HF")).setup(
                pretrained="example/model",
                auto_class="AutoModelForCausalLM",
                training=False,
            )

        self.assertIs(context.data["model.instance"], model)
        self.assertFalse(model.training)

    def test_diffusers_pipeline_is_loaded_and_moved(self):
        from starling_ml.integrations.diffusers import DiffusionPipelineLoader

        fake = types.ModuleType("diffusers")
        pipeline = types.SimpleNamespace()
        pipeline.to = lambda device: setattr(pipeline, "device", device) or pipeline

        class DiffusionPipeline:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                pipeline.name = name
                return pipeline

        fake.DiffusionPipeline = DiffusionPipeline
        context = Context({}, {"Diffusion": {"creates": ["diffusion.pipeline"]}})
        with patch.dict(sys.modules, {"diffusers": fake}):
            DiffusionPipelineLoader(context.view("Diffusion")).setup(
                pretrained="example/diffusion", device="cpu"
            )

        self.assertIs(context.data["diffusion.pipeline"], pipeline)
        self.assertEqual(pipeline.device, "cpu")

    def test_nnunet_predictor_is_initialized_once(self):
        from starling_ml.integrations.nnunet import NNUNetPredictorLoader

        predictor_module = types.ModuleType(
            "nnunetv2.inference.predict_from_raw_data"
        )

        class Predictor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def initialize_from_trained_model_folder(self, folder, **kwargs):
                self.folder = folder
                self.initialize_kwargs = kwargs

        predictor_module.nnUNetPredictor = Predictor
        fake_modules = {
            "nnunetv2": types.ModuleType("nnunetv2"),
            "nnunetv2.inference": types.ModuleType("nnunetv2.inference"),
            "nnunetv2.inference.predict_from_raw_data": predictor_module,
        }
        context = Context({}, {"NNUNet": {"creates": ["nnunet.predictor"]}})
        with patch.dict(sys.modules, fake_modules):
            NNUNetPredictorLoader(context.view("NNUNet")).setup(
                trained_model_folder="model_folder",
                device="cpu",
            )

        predictor = context.data["nnunet.predictor"]
        self.assertEqual(predictor.folder, "model_folder")
        self.assertEqual(predictor.initialize_kwargs["use_folds"], (0,))


if __name__ == "__main__":
    unittest.main()
