"""Минимальный пользовательский запуск после pip install."""
from starling_ml import Engine, get_config, analyze_config

config = get_config('weighted_segmentation', max_steps=5)
config['modules']['optimizer']['params']['lr'] = 3e-4
report = analyze_config(config)
print(report)
report.raise_for_errors()
Engine(config).run()
