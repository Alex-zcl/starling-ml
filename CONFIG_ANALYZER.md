# Анализатор конфигурации

```python
from starling_ml import analyze_config
report = analyze_config(config)
print(report)
report.raise_for_errors()
```

Report имеет `issues`, `errors`, `warnings`, `objectives`, `ok`. Каждый Issue содержит
severity, code, location и русское сообщение. Это позволяет показать пользователю
обычный текст либо встроить диагностику в GUI. Предупреждения не блокируют запуск.

## Три стадии

| Когда | Что проверяется | Что не выполняется |
|---|---|---|
| analyze_config(config) | разделы, class/setup signatures, contracts, owners, setup cycles, references, DSL, известные настройки losses | setup, data loading, forward |
| metadata=statistics | class mapping, background, counts, split, sampling changes, include_classes | пересчёт dataset statistics |
| batch=context_mapping | опубликованный scalar/finite loss, prediction dtype и device совместимость | потребление следующего batch, optimizer step |

Формы targets/masks дополнительно проверяются непосредственно pure loss functions
при первом вызове. Анализатор не доказывает корректность пользовательского Python-кода
и не гарантирует завершение произвольного while. Для while задавайте max_iterations.
`**kwargs` стороннего wrapper проверяет соответствующая библиотека при setup.

Engine автоматически проверяет структурные ошибки. Полный текст warnings печатает
пользователь через report; анализатор не меняет weights, reducers или порядок pipeline.
Для dataset-derived metadata после setup вызовите анализ ещё раз с data.statistics.

## Основные диагностики

| Код | Уровень | Последствие |
|---|---|---|
| UNKNOWN_KEY / MODULE_KEY | error | опечатка в конфиге |
| SETUP_SIGNATURE | error | неизвестный/пропущенный setup parameter или import path |
| DUPLICATE_OWNER / NO_OWNER | error | неоднозначное владение либо отсутствующий producer |
| SETUP_PERMISSION / RUNTIME_KEY | error | ссылка не соответствует контракту |
| SETUP_CYCLE | error | циклическая инициализация |
| REDUCTION / NORMALIZATION / TASK_MODE / EMPTY_POLICY | error | неподдерживаемая математика |
| CLASS_MAPPING / COUNT_SHAPE / COUNTS / BACKGROUND | error | metadata несовместимы |
| SCALAR_LOSS / FINITE_LOSS / DEVICE / DTYPE | error | неверный опубликованный runtime result |
| WEIGHT_CANCELLATION | warning | одинаковые importance сокращаются в weighted mean |
| OVERLAP_CANCELLATION | warning | общий multiplier TP/FP/FN сокращается без smoothing |
| PRESENCE_NORMALIZATION | warning | present/empty weight не задаёт абсолютную силу штрафа |
| EMPTY_IGNORED | warning | empty weights/cost не действуют при ignore |
| BATCH_STATS | warning | объединённый overlap отличается от mean sample Dice |
| MULTIPLE_COMPENSATION | warning | несколько dataset frequency controllers могут усиливать один дисбаланс |
| STRONG_COMPENSATION / MISSING_CLASS | warning | большая gamma либо политика отсутствующих классов требует осмысления |
| LEGACY_FREQUENCY | warning | старый epsilon-based controller скрывает zero counts |
| MICRO_MEAN | warning | разные denominators microbatches дают другое среднее |
| EXACT_MEMORY | warning | сохранение всех graphs требует дополнительной памяти |
| TERM_NORMALIZATION | warning | общий scale term weights сокращается |
| MUTATES | warning | read reference не описывает изменение модели по ссылке |
| NO_BACKGROUND | warning | softmax без background не выбирает «ни один класс» |
| STATISTICS_SPLIT / SAMPLING_DISTRIBUTION | warning | источник частот отличается от training distribution |
| UNBOUNDED_LOOP | warning | нет аварийного max_iterations |
| METADATA / RUNTIME / LOSS_OUTPUT | pending | требуется следующий этап проверки |

Сводка objectives показывает reducer, тип denominator и источники coefficients.
Для прямого DatasetWeights источника указаны level, balance, unit и gamma.
Прослеживание длинных пользовательских цепочек producers не является символическим
анализом: составные WeightProduct и custom controllers требуют проверки человеком.
Контроллер сам не решает, полезно ли одновременно исправлять pixel и study imbalance.

Пример разумной конфигурации с предупреждением: Dice имеет empty_target=false_positive,
empty_weight=.625, present_weight=2.5, empty_cost=.2. Presence weights балансируют группы,
а cost ослабляет пустой штраф даже в полностью пустом batch. Предупреждение о сокращении
importance описывает поведение, а не объявляет конфигурацию ошибочной.
