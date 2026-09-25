"""Стандартный запуск 0.4 с консолью, tqdm, текстом и TensorBoard."""

from starling_ml import Engine, get_experiment_config, analyze_config

monitor_scalars = {
    "loss": "train.loss",
    "focal": "loss.pointwise",
    "dice": "loss.dice",
}

# tracking = console + tqdm + experiment.log + TensorBoard. ClearML включается
# только отдельным явным профилем, потому что создаёт внешнюю задачу.
config = get_experiment_config(
    "weighted_segmentation",
    max_steps=5,
    monitoring="tracking",
    monitoring_options={
        "run_name": "weighted_segmentation",
        "scalars": monitor_scalars,
    },
)
config["modules"]["optimizer"]["params"]["lr"] = 3e-4

report = analyze_config(config)
print(report)
report.raise_for_errors()

engine = Engine(config).run()
print(engine)
