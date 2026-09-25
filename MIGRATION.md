# Переход на 0.4

## 0.3 → 0.4

1. Старый `get_config()` и ручное добавление modules продолжают работать.
2. Для готового запуска используйте `get_experiment_config(..., monitoring=...)` или
   низкоуровневый `compose(recipe(...), monitoring_profile(...))`.
3. Новые Python-конфиги используют RunManager только для global step/stop, а переходы
   train/validation принадлежат PhaseManager. Legacy-поля `run.phase`,
   `run.validation_due` и старые lifecycle signals временно публикуются для совместимости.
4. Универсальные события: `phase_started`, `phase_step_end`, `phase_ended`,
   `phase_completed`; текущее состояние находится в `phase.name/index/step/cycle/due`.
5. Stateful metrics/losses следует создавать отдельно на фазу. `phase_module()` умеет
   развернуть `{phase}`-шаблон; validation class/params можно переопределить.
6. Профили логирования не входят в task recipe и не влияют на objective. `standard` —
   console/progress/text, `tracking` добавляет TensorBoard, ClearML выбирается явно.
7. `segmentation_validation` теперь публикует отдельные train/validation metrics и
   `validation.loss`; это добавляет ключи, но не меняет train objective.

## 0.2 → 0.3

До первого публичного релиза исправлено рабочее название: distribution
`starlilng-ml` → `starling-ml`, import `starlilng` → `starling_ml`. Старый import
не публиковался в PyPI и compatibility alias намеренно не добавляется, чтобы опечатка
не стала частью постоянного API.

1. Предпочтительный API — `from starling_ml import Engine, get_config, analyze_config`.
   Старый `load_engine(directory)` и четыре YAML сохранены.
2. Новые binary/overlap модули по умолчанию используют pair_mean; legacy pure
   `reduce_pointwise` сохраняет sample_mean. Старые результаты могут измениться,
   если число валидных классов между samples неодинаково. Укажите reducer явно.
3. Positive/negative binary coefficients — costs без автоматической нормировки.
   Alias positive_weight/negative_weight в BCE сохранён; новые конфиги используют cost.
   В overlap positive_weight/negative_weight означают внутренние weights TP/FP/FN.
4. Внешние present_weight/empty_weight появились непосредственно в overlap API.
   PresenceWeights + case_class_weight остаётся доступным. Не применяйте оба способа
   для одной и той же частотной компенсации случайно.
5. CE label_weight — совместимый alias label_cost, а не PyTorch weighted-mean denominator.
   Нормированную importance задавайте target-class element map. Focal p берётся из logits.
6. Default OptimizationManager accumulation_mode теперь exact. Для внешнего loss scalar
   задайте normalizer key с настоящим denominator или выберите micro_mean явно.
   Native losses и Mixer сохраняют Objective stats. Exact удерживает graphs до backward.
7. Checkpoint теперь полный; старый weights-only формат доступен через full_state=False.
   Legacy snapshots не восстанавливают полный 0.3 run state. Сохраняйтесь на optimizer boundary.
8. После run Engine закрыт; для resume создайте новый Engine и load_checkpoint.
9. Контракты поддерживают mutates. Старые reads-only model contracts получают предупреждение,
   не автоматическое изменение прав.
10. Для DatasetStats указывайте sample_unit='patch' при нарезке patches. Negative patch
    не является доказательством negative study. Legacy FrequencyWeights остаётся, но
    предпочтителен DatasetWeights с явной missing policy.
11. Optional dependencies не нужны для обычного import; detection examples требуют scipy.
    Стандартные конфиги находятся внутри Python-пакета и доступны после установки wheel.

Новая нормировка — намеренное изменение objective, не только rename параметров.
Перед переносом важного эксперимента сравните loss и gradients на фиксированном batch.
