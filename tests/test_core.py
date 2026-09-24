"""Проверки Context и общего pipeline DSL."""

import unittest
from pathlib import Path

from starling_ml.core.config import load_engine
from starling_ml.core.context import Context
from starling_ml.core.engine import Engine


class ContextTests(unittest.TestCase):
    def test_permissions(self):
        context = Context(
            {},
            {
                "A": {"creates": ["x"]},
                "B": {"reads": ["x"]},
            },
        )
        a = context.view("A")
        b = context.view("B")
        a["x"] = 3
        self.assertEqual(b["x"], 3)
        with self.assertRaises(PermissionError):
            b["x"] = 4

    def test_duplicate_owner_is_rejected(self):
        with self.assertRaises(ValueError):
            Context({}, {"A": {"creates": ["x"]}, "B": {"creates": ["x"]}})


class PipelineTests(unittest.TestCase):
    def test_step_driven_validation(self):
        engine = Engine(
            constants={},
            modules={
                "Run": {
                    "class": "starling_ml.modules.control.run.RunManager",
                    "params": {
                        "max_steps": 5,
                        "validate_every": 2,
                        "validate_at_end": True,
                    },
                }
            },
            contracts={
                "Run": {
                    "creates": [
                        "run.step",
                        "run.phase",
                        "run.validation_index",
                        "run.validation_due",
                        "run.stop",
                    ]
                }
            },
            pipeline=[
                {"signal": "run_started"},
                {
                    "while": {
                        "condition": "$ctx:run.stop",
                        "equals": False,
                        "body": [
                            {"signal": "step_completed"},
                            {
                                "when": {
                                    "condition": "$ctx:run.validation_due",
                                    "equals": True,
                                    "body": [{"signal": "validation_completed"}],
                                }
                            },
                        ],
                    }
                },
            ],
        ).setup().run()

        self.assertEqual(engine.context.data["run.step"], 5)
        self.assertEqual(engine.context.data["run.validation_index"], 3)
        self.assertEqual(engine.context.data["run.phase"], "finished")
        self.assertTrue(engine.context.data["run.stop"])


    def test_zero_steps_finish_immediately(self):
        # Нулевой лимит должен сразу дать согласованные stop и finished,
        # иначе reporter-ы увидят противоречивое состояние запуска.
        engine = Engine(
            constants={},
            modules={
                "Run": {
                    "class": "starling_ml.modules.control.run.RunManager",
                    "params": {"max_steps": 0},
                }
            },
            contracts={
                "Run": {
                    "creates": [
                        "run.step",
                        "run.phase",
                        "run.validation_index",
                        "run.validation_due",
                        "run.stop",
                    ]
                }
            },
            pipeline=[{"signal": "run_started"}],
        ).setup().run()

        self.assertEqual(engine.context.data["run.step"], 0)
        self.assertEqual(engine.context.data["run.phase"], "finished")
        self.assertTrue(engine.context.data["run.stop"])

    def test_shipped_demo_config_is_valid(self):
        # Проверяем именно поставляемые YAML: так забытая setup-зависимость
        # не пройдёт unit-тесты и не обнаружится только у пользователя.
        project_root = Path(__file__).resolve().parents[1]
        engine = load_engine(project_root / "configs" / "segmentation_demo")
        engine.setup()
        self.assertIn("model.instance", engine.context.data)

    def test_setup_context_reference_requires_read_permission(self):
        with self.assertRaises(ValueError):
            Engine(
                constants={},
                modules={
                    "A": {"class": "starling_ml.core.module.Module", "params": {}},
                    "B": {
                        "class": "starling_ml.core.module.Module",
                        "params": {"value": "$ctx:x"},
                    },
                },
                contracts={
                    "A": {"creates": ["x"]},
                    "B": {},
                },
                pipeline=[],
            )

    def test_iterate_uses_stop_iteration(self):
        engine = Engine(
            constants={},
            modules={
                "Source": {
                    "class": "tests.helpers.FiniteSource",
                    "params": {"limit": 3},
                },
                "Sink": {
                    "class": "tests.helpers.SumSink",
                    "params": {},
                },
            },
            contracts={
                "Source": {"creates": ["item"]},
                "Sink": {"reads": ["item"], "creates": ["total"]},
            },
            pipeline=[
                {
                    "iterate": {
                        "source": "Source",
                        "body": [{"call": "Sink"}],
                    }
                }
            ],
        ).setup().run()
        self.assertEqual(engine.context.data["total"], 3)


if __name__ == "__main__":
    unittest.main()
