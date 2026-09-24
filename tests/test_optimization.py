"""Проверка, что accumulation отделён от global step."""

import unittest

import torch

from starling_ml.core.context import Context
from starling_ml.modules.optimization import OptimizationManager


class OptimizationTests(unittest.TestCase):
    def test_accumulation_sets_did_step_only_on_real_update(self):
        contracts = {
            "Model": {"creates": ["model.instance"]},
            "Loss": {"creates": ["train.loss"]},
            "Optimization": {
                "reads": ["model.instance", "train.loss"],
                "creates": [
                    "optim.main.optimizer",
                    "optim.main.did_step",
                    "optim.main.step",
                    "optim.main.micro_step",
                    "optim.main.grad_norm",
                    "optim.main.skipped",
                ],
            },
        }
        context = Context({}, contracts)
        model = torch.nn.Linear(1, 1, bias=False)
        context.view("Model")["model.instance"] = model

        manager = OptimizationManager(context.view("Optimization"))
        manager.name = "Optimization"
        manager._emit = lambda *args, **kwargs: None
        manager.setup(model=model, accumulation=2, accumulation_mode="micro_mean", lr=0.1)

        for index in range(2):
            prediction = model(torch.ones(1, 1))
            context.view("Loss")["train.loss"] = prediction.square().mean()
            manager()
            if index == 0:
                self.assertFalse(context.data["optim.main.did_step"])

        self.assertTrue(context.data["optim.main.did_step"])
        self.assertEqual(context.data["optim.main.step"], 1)
        self.assertEqual(context.data["optim.main.micro_step"], 2)


if __name__ == "__main__":
    unittest.main()
