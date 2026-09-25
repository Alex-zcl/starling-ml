"""Публичные сценарии композиции, фаз и monitoring версии 0.4."""

from pathlib import Path
import tempfile
import unittest

from starling_ml import (
    ConfigConflict,
    Engine,
    analyze_config,
    compose,
    fragment,
    get_config,
    get_experiment_config,
    monitoring_profile,
    phase_module,
    recipe,
)
from starling_ml.core.context import Context
from starling_ml.modules.monitoring.progress import TqdmReporter


class CompositionTests(unittest.TestCase):
    def test_recipe_and_monitoring_are_composable(self):
        config = compose(
            recipe("classification", max_steps=1),
            monitoring_profile("console"),
        )
        self.assertIn("console_logger", config["modules"])
        self.assertTrue(analyze_config(config).ok)
        self.assertEqual(Engine(config).run().context.data["run.step"], 1)

    def test_conflict_requires_explicit_replacement(self):
        first = fragment(
            "first",
            modules={"x": {"class": "starling_ml.core.module.Module", "params": {}}},
            contracts={"x": {}},
        )
        second = fragment(
            "second",
            modules={"x": {"class": "starling_ml.core.module.Module", "params": {}}},
            contracts={"x": {}},
        )
        with self.assertRaises(ConfigConflict):
            compose(first, second)

    def test_phase_module_expands_independent_instances(self):
        part = phase_module(
            "loss",
            "starling_ml.modules.losses.basic.MSELoss",
            params={
                "prediction": "{phase}.prediction",
                "target": "{phase}.target",
                "output": "loss.{phase}.total",
            },
            contract={
                "reads": ["{phase}.prediction", "{phase}.target"],
                "creates": ["loss.{phase}.total"],
            },
            overrides={
                "validation": {
                    "class": "starling_ml.modules.losses.basic.L1Loss",
                }
            },
        )
        self.assertEqual(set(part.config["modules"]), {"train_loss", "validation_loss"})
        self.assertEqual(
            part.config["modules"]["validation_loss"]["class"],
            "starling_ml.modules.losses.basic.L1Loss",
        )
        self.assertEqual(
            part.config["contracts"]["train_loss"]["creates"],
            ["loss.train.total"],
        )


class PhaseAndMonitoringTests(unittest.TestCase):
    def test_validation_recipe_has_separate_phase_metrics_and_loss(self):
        engine = Engine(get_config("segmentation_validation", max_steps=3)).run()
        data = engine.context.data
        self.assertEqual(data["run.step"], 3)
        self.assertEqual(data["run.validation_index"], 3)
        self.assertEqual(data["phase.name"], "finished")
        self.assertIn("metrics.train.dice_mean", data)
        self.assertIn("metrics.validation.dice_mean", data)
        self.assertIn("validation.loss", data)

    def test_ready_experiment_writes_text_log(self):
        with tempfile.TemporaryDirectory() as directory:
            config = get_experiment_config(
                "classification",
                max_steps=1,
                monitoring="text",
                monitoring_options={"root": directory, "run_name": "smoke"},
            )
            self.assertTrue(analyze_config(config).ok)
            Engine(config).run()
            log = Path(directory) / "smoke" / "experiment.log"
            self.assertTrue(log.is_file())
            self.assertIn("loss=", log.read_text(encoding="utf-8"))

    def test_tqdm_cleanup_is_safe_before_setup(self):
        context = Context({}, {"progress": {}})
        TqdmReporter(context.view("progress")).close()


if __name__ == "__main__":
    unittest.main()
