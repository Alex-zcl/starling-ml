# Starling ML 0.3.0

[![CI](https://github.com/Alex-zcl/starling-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Alex-zcl/starling-ml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/starling-ml.svg)](https://pypi.org/project/starling-ml/)
[![Python](https://img.shields.io/pypi/pyversions/starling-ml.svg)](https://pypi.org/project/starling-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Alex-zcl/starling-ml/blob/main/LICENSE)

Небольшой конфигурируемый движок для PyTorch. Пользователь импортирует Engine,
получает обычный словарь стандартного конфига, меняет параметры и запускает его.
Математика, загрузка данных, оптимизация и мониторинг остаются отдельными модулями.

## Быстрый запуск

Python >= 3.10; основная проверка выполнена на Python 3.12, PyTorch CPU.

```bash
python -m pip install starling-ml
```

```python
from starling_ml import Engine, get_config, analyze_config

config = get_config("weighted_segmentation", max_steps=5)
config["modules"]["optimizer"]["params"]["lr"] = 3e-4

report = analyze_config(config)
print(report)
report.raise_for_errors()
engine = Engine(config).run()
```

Стандартные конфиги используют маленькие синтетические datasets и модели.
Они проверяют исполнимость сценария, но не заменяют подготовку реального dataset.
Никакие модели не скачиваются. Для работы с detection-рецептами установите
`python -m pip install 'starling-ml[detection]'`.

CLI использует тот же API:

```bash
starling-ml --recipe weighted_segmentation --analyze-only
starling-ml --recipe segmentation_validation --steps 3
starling-ml --config experiment.yaml
```

Python-модуль называется `starling_ml`: имя `starling` уже занято другим
проектом в PyPI, поэтому Starling ML не захватывает чужое пространство импортов.

## Конфигурация

Четыре основные секции: `constants`, `modules`, `contracts`, `pipeline`.
Необязательная `metadata` описывает задачу, class mapping, background и statistics.
`Engine(config)` принимает dict, путь к единому YAML или каталог старых четырёх YAML.
`get_config()` каждый раз возвращает независимый словарь.

```python
from starling_ml import get_config, save_config, Engine
config = get_config("classification")
save_config(config, "experiment.yaml")
Engine("experiment.yaml").run()
```

Ссылки `$const:name` и `$ctx:key` разрешаются в setup. Runtime-параметры вроде
`prediction="model.output"` обозначают context keys и читаются при вызове модуля.
Контракт объявляет `creates`, `reads`, `updates` и `mutates`. Setup сортируется по
зависимостям `$ctx`; runtime pipeline выполняется в записанном пользователем порядке.
Конфиги с пользовательскими import paths должны быть доверенными Python-конфигами.

## Стандартные сценарии

| Имя get_config | Исполняемый пример |
|---|---|
| classification | MLP + multiclass CE |
| regression | MLP + MSE |
| segmentation | 2D softmax CE + Dice |
| segmentation3d | 3D softmax CE + Dice |
| multilabel | sigmoid BCE + Dice |
| weighted_segmentation | dataset pixel costs для focal, study weights и ручные class priorities для Dice |
| segmentation_validation | полный train/validation lifecycle с накоплением confusion stats |
| language_model | causal GRU, token logits `[B,T,C]`, CE с class_dim=-1 |
| diffusion | DDPM forward noise process + noise predictor |
| contrastive | две views, symmetric InfoNCE |
| distillation | student/teacher KL с temperature |
| rl_bandit | categorical policy gradient для contextual bandit |
| gan | discriminator/generator, два optimizer, detach и freeze |
| detection | query model, Hungarian matching, boxes; требует scipy |
| instance_segmentation | detection baseline с matched masks; требует scipy |

Для последних двух: `python -m pip install '.[detection]'`.
Внешние SMP, MONAI, Transformers, Diffusers, nnU-Net и логгеры подключаются лениво.
Матрица реальных возможностей и границ — в `INTEGRATIONS.md`.

## Подключение своих данных и моделей

В стандартном конфиге замените factory и kwargs модуля `dataset` на import path
вашего map-style dataset; каждая запись должна иметь `input` и `target` (или измените
`batch.params.outputs`). `BatchSource` поддерживает batch size, shuffle, seed,
cycle, drop_last и пользовательский `collate_fn` callable/import path.
Для ragged targets используйте явный collate: padding никогда не добавляется скрыто.
Затем замените `model.params.factory` и `model.params.kwargs` на свой torch.nn.Module.

`PrepareModel` переносит модель на устройство до создания optimizer; модули
`MoveToDevice` переносят вложенные batches, сохраняя целочисленные labels.
При замене outputs обновляйте также contracts и параметры consumers.

## Проверка конфигурации

`analyze_config()` возвращает ошибки, предупреждения, непроверенные условия и
сводку objectives. `Engine` автоматически блокирует структурные ошибки; предупреждения
пользователь печатает через report. После setup доступны dataset metadata:

```python
engine = Engine(config).setup()
report = analyze_config(config, metadata=engine.context.data.get("data.statistics"))
print(report)
report.raise_for_errors()
engine.run()
```

`batch=engine.context.data` дополнительно проверяет опубликованные runtime outputs.
Анализатор не потребляет batch, не делает optimizer step и не меняет формулу.
Полное описание предупреждений — в `CONFIG_ANALYZER.md`.

## Checkpoint и параллельность

```python
from starling_ml.checkpoint import save_checkpoint, load_checkpoint
save_checkpoint(engine, "checkpoint.pt")
config["constants"]["max_steps"] = 10
resumed = load_checkpoint(Engine(config), "checkpoint.pt").run()
```

Восстанавливаются состояния модулей, моделей, optimizer, scaler, counters, controllers,
RNG и cursor встроенного BatchSource. Сохранять можно только на границе optimizer step.
Чужие checkpoints не загружайте: полный training state использует Python serialization.

Exact accumulation объединяет достаточные stats objective и сохраняет графы всех
microbatches до backward. Это требует больше памяти. Старое усреднение microbatch losses
доступно как `accumulation_mode="micro_mean"`. Для внешнего scalar в exact режиме
нужен явный normalizer или собственный Objective. Подробности — в `LOSS_WEIGHTING.md`.

`examples/train_distributed.py` — DDP через torchrun. `examples/check_ddp.py` — проверка
loss и gradients на двух Gloo ranks. В текущей среде сокеты запрещены, поэтому DDP
**не отмечен как прошедший проверку**. `launcher.run_experiments` запускает независимые
эксперименты в spawn-процессах; этот путь проверен. Каждый config задаёт собственный
seed/device; launcher разделяет рабочие каталоги и outputs.

## Документы для передачи следующему разработчику

Начать с `HANDOFF.md`, затем `PHILOSOPHY.md`, `LOSS_WEIGHTING.md`, `CONFIG_ANALYZER.md`.
`MIGRATION.md` описывает отличия от 0.2. `VALIDATION_REPORT.md` отделяет выполненные
проверки от ограничений. `RESUME_NOTES.md` фиксирует состояние и следующие работы.
`AUDIT_v0.2.0.md` и `WEIGHTING_DESIGN.md` — исторические материалы, не описание API.

## Разработка и безопасность

Инструкции для изменений находятся в `CONTRIBUTING.md`, история релизов — в
`CHANGELOG.md`. Проект распространяется по лицензии MIT. Конфиги содержат Python
import paths, а checkpoints используют PyTorch serialization: загружайте их только
из доверенных источников. Уязвимости следует сообщать через GitHub private
vulnerability reporting, как описано в `SECURITY.md`.
