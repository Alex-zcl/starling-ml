"""Запускает небольшой step-driven segmentation experiment."""

from pathlib import Path

from starling_ml import load_engine


if __name__ == "__main__":
    root = Path(__file__).parent
    engine = load_engine(root / "configs" / "segmentation_demo")
    engine.setup().run()

    # Финальный вывод оставлен явным, чтобы smoke-test было легко проверить.
    print("final step:", engine.context.data["run.step"])
    print("validation runs:", engine.context.data["run.validation_index"])
    print("final validation dice:", engine.context.data["metrics.validation.dice_mean"])
    print("dice loss weight:", engine.context.data["weights.loss.dice"])
