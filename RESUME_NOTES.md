# Состояние для возобновления — 0.3.0

Дата фиксации: 23 сентября 2026. Это документ состояния **сохранённых файлов**.
При возобновлении в workspace нашлись частичные 0.3 исходники и старые документы,
но не ранее заявленные анализатор/standard configs/78 tests. Недостающее восстановлено
и проверено заново; сообщения прежнего чата не являются evidence выполнения тестов.

## Готово

- Public Engine(config), load/save YAML, get_config, analyze_config, CLI.
- Dataset counts, шесть ролей весов, explicit masks/costs/importance.
- BCE/focal/CE, overlap reducers, Generalized Dice, Mixer sufficient stats.
- Stable sums, ignored targets, [B]/[B,C]/tokens/2D/3D shapes.
- Context mutates/frozen constants, lifecycle guards, weight signals.
- Optimizer exact/micro_mean, clipping, partial-window policy, state.
- Full checkpoints с RNG, module states и встроенным BatchSource cursor.
- Device transforms, factories, scheduler, early stopping, EMA.
- DDP integration и independent launcher; статусы проверок различаются.
- 15 synthetic recipes, включая weighted segmentation, validation, GAN и ragged detection.
- Документы для самостоятельной передачи проекта другой модели.

## Последняя проверка

См. VALIDATION_REPORT.md и validation/tests.txt. В отчёт включены только фактически
выполненные проверки. DDP Gloo остановился до обучения из-за Operation not permitted
при создании socket. CUDA отсутствует. Optional pretrained pipelines не запускались.

## Ограничения и технические решения

Exact accumulation хранит graphs: это точная математика reduction, но не экономия памяти.
Перед checkpoint pending accumulation запрещён; flush/drop задаются явно сигналом
flush_optimization и политикой partial. Для нескольких optimizer run.step меняется
только после выбранного логического update в pipeline.

BatchSource — deterministic map-style cursor. Multi-worker DataLoader, случайный
preprocessing во внешних процессах, streaming datasets и environment states требуют
собственного checkpoint protocol. Per-rank restore требует прежнего world size.

Synthetic validation использует тот же dataset для проверки wiring; для реального
эксперимента нужны отдельные train/validation datasets. Synthetic language target
показывает layout, а не production language modeling labels policy.

VLM/Florence, latent/text-conditioned diffusion и полный PPO rollout не доведены до
реального end-to-end recipe. Detection baseline не является COCO evaluator.

## Карта кода

- starling_ml/core: Context, Module, Engine, config I/O.
- starling_ml/analysis.py: диагностика без изменения config.
- starling_ml/configs.py и recipes.py: импортируемые стандартные конфиги.
- starling_ml/ops: чистая математика; objectives.py — sufficient stats.
- starling_ml/modules: runtime adapters и stateful modules.
- starling_ml/checkpoint.py: RNG/state I/O вне Engine.
- starling_ml/distributed.py: process group + DDP adapter.
- starling_ml/launcher.py: независимые процессы с отдельными outputs.
- tests/test_v03_math.py, test_v03_runtime.py: regression проверки новой версии.
- examples/check_ddp.py: воспроизводимая граница непроверенного distributed пути.

При продолжении сначала прочитать HANDOFF.md. Не пересобирать архитектуру с нуля
и не менять математические договорённости ради совместимости с привычным Trainer.
