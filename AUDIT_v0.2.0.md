# Аудит Starling ML 0.2.0

## Вывод

Движок в основном соответствует своей философии на уровне архитектуры: небольшое ML-независимое ядро, явный runtime-pipeline, отдельные loss-модули и тонкие интеграции. Однако часть заявленных гарантий пока сильнее реализации. Контракты защищают присваивание ключей, но не изменение объектов по ссылке; обработчики событий могут зависеть от порядка YAML; часть собственной математики остаётся внутри runtime-модулей.

Штатная демонстрация работоспособна. Вместе с тем дополнительные проверки выявили ошибки в численной устойчивости, редукции масок и lifecycle. Перед расширением библиотеки задач стоит исправить эти основания.

Это аудит исходной версии, а не исправленный релиз. Исходные файлы проекта не изменены.

## Объект и границы проверки

- Архив: `starling_ml_v0.2.0(1).zip`.
- SHA-256: `57201e051200d10f7631fa2dcd62df2b4566e780ca48a1dab4e061db86629385`.
- Основной нормативный документ: русская `PHILOSOPHY.md` внутри архива. Отдельно приложенная `PHILOSOPHY (1).md` совпадает с ней побайтово.
- Приложенные README, RESUME_NOTES, LOSS_WEIGHTING и STEP_LIFECYCLE также совпадают с соответствующими файлами архива.
- Английская `PHILOSOPHY(1).md` описывает более ранний вариант: например, две политики пустых масок вместо трёх. Её нельзя использовать как единственную спецификацию 0.2.0.
- Прочитаны core, ops, runtime-модули, интеграции, конфигурации и штатные тесты.
- Исполнение: Python 3.12, PyTorch `2.14.0+cpu`, CPU, один OpenMP-поток. PyTorch установлен отдельно для аудита; зависимости проекта не менялись.
- CUDA, реальное distributed training и end-to-end запуск тяжёлых optional integrations не проверялись. Проверка MONAI ниже относится к границе адаптера с подменённым inferer.

## Что прошло

Команда `python -m unittest discover -s tests -v`: **23 теста, все прошли**.

`python run_demo.py`:

| Показатель | Результат |
|---|---:|
| Финальный global step | 24 |
| Число validation-запусков | 3 |
| Финальный validation Dice | 0.663607656955719 |
| Финальный коэффициент Dice loss | 0.5 |

Demo подтверждает работоспособность конкретной CPU multilabel segmentation-композиции. Его результат не доказывает корректность всех форм тензоров, режимов weighting и последовательностей событий.

## Соответствие философии

| Принцип | Оценка | Основание |
|---|---|---|
| Engine управляет порядком, не ML-семантикой | Соблюдён | В `core` нет предметной логики обучения; DSL состоит из общих управляющих конструкций. |
| Setup-граф автоматический, runtime-порядок явный | Соблюдён | `_setup_order()` использует `$ctx:`, `_run_steps()` исполняет указанный порядок. |
| Один создатель каждого ключа | Соблюдён для ключей | Дубли владельцев и неизвестные reads/updates отклоняются. |
| Управляемый доступ к состоянию | Частично | Присваивания проверяются, но полученные mutable-объекты можно менять без `updates`. |
| Константы неизменяемы | Частично | `MappingProxyType` защищает только верхний уровень. |
| Чистая математика в ops | Частично | Overlap, reduction, weights вынесены; focal, binary cost weights и detection-statistics остаются в модулях. |
| LossMixer только смешивает losses | Соблюдён | Activation и task-specific weighting отсутствуют. Проверку scalar loss стоит усилить. |
| Контроллер создаёт вес, loss применяет | Соблюдён структурно | Но `WeightProduct` и producers могут читать/писать на одном событии в порядке YAML. |
| Сигналы сообщают события, Context хранит tensors | В основном соблюдён | Основные tensors находятся в Context; проблемы связаны с порядком вложенных событий. |
| Наблюдение отделено от control flow | Соблюдён | Logger, checkpoint, progress не запускают optimizer или validation. |
| Интеграции тонкие и ленивые | Соблюдён структурно | Отсутствие optional packages не мешает базовым импортам с установленным torch. Реальные сценарии ещё требуют проверки. |

Наличие model/optimizer в Context само по себе не считаю нарушением: документы явно предусматривают `model.instance`. Важнее определить права на изменение таких объектов и не помещать orchestration внутрь Context.

## Подтверждённые ошибки и дефекты

Приоритет P1 означает риск тихо неправильного обучения либо некорректного завершения; P2 — ошибка ограниченного режима или важная проблема композиции. Сценарии ниже воспроизведены отдельно от штатных тестов.

### E1. FP16-редукции дают NaN — P1

Места: `starling_ml/ops/reduction.py`, `weighted_mean()`, строки 8–22; `starling_ml/ops/losses.py`, `_overlap_stats()`.

Взвешенные суммы считаются в dtype входа. В FP16 сумма 65 536 единиц переполняется. Деление двух переполненных сумм даёт NaN. Отдельно `eps=1e-12` не представим в FP16: при нулевых весах скрытая ветка делит 0 на 0, и `torch.where` не устраняет NaN в backward.

Воспроизведено:

- `weighted_mean(ones[1,1,256,256], ones, dim=(2,3))` в FP16 → **NaN**, ожидается 1.
- Overlap на FP16-маске 256×256, prediction=0.5, target=1 → **NaN**.
- `reduce_pointwise()` на FP16 с полностью нулевыми element weights → forward равен 0, **градиенты NaN**.

Исправление: считать суммы и отношения как минимум в FP32 для FP16/BF16; формировать безопасный ненулевой знаменатель до деления; проверять backward полностью исключённой маски. GradScaler не исправляет NaN, уже возникший при вычислении loss.

### E2. Ignore-index не исключает полностью игнорируемый sample из финального среднего — P1

Места: `ops/reduction.py`, `reduce_per_sample()`, строки 74–85; `modules/losses/basic.py`, CrossEntropy и MulticlassFocalLoss.

Внутри sample ignore mask учитывается, но полностью исключённый sample превращается в нулевой loss и остаётся в среднем по batch. Для классификации `[B,C]` маска вообще не применяется на этапе `reduce_per_sample()`, потому что нет spatial dimensions.

Воспроизведено на нулевых logits для двух классов:

| Сценарий | Получено | Ожидается после исключения ignored sample |
|---|---:|---:|
| CE, targets `[0,-100]` | 0.346574 | 0.693147 |
| CE, один валидный sequence и один полностью ignored | 0.346574 | 0.693147 |
| Multiclass focal, targets `[0,-100]`, gamma=2 | 0.086643 | 0.173287 |

Следствие: масштаб loss меняется от числа полностью неразмеченных примеров. Аналогичная проблема с полностью нулевыми element masks возможна в pointwise reduction: `valid_sample` выводится из class-level weights, а не из наличия валидных элементов.

Исправление: сохранять validity/массу валидных элементов через все стадии редукции и исключать пустые группы из соответствующих знаменателей. При этом явно сохранить выбранную семантику sample-average или element-average.

### E3. Element weights теряются на тензорах `[B,C]` — P1

Место: `ops/reduction.py`, `reduce_pointwise()`, строки 46–51.

Если spatial dimensions отсутствуют, код возвращает `loss` без применения `element_weight`.

Пример: `loss=[[1,100]]`, `element_weight=[[1,0]]` → **50.5**, а masked mean должен быть 1.

Затрагивает multilabel classification, маски неизвестных labels, BCE/focal с positive/negative weights на `[B,C]`, многомерную regression без spatial dimensions.

Исправление: определить явную семантику элемента `[b,c]` и передавать его mask/cost в последующую редукцию; не пропускать weighting из-за отсутствия spatial осей.

### E4. Запрос остановки из обработчика события не останавливает дальнейший lifecycle — P1

Места: `modules/control/run.py`, `_on_step_completed()` и `_on_validation_completed()`; `core/engine.py`, `signal()`.

Сигналы синхронны и вызываются вложенно. Другой модуль может вызвать `stop_requested` во время `train_step_end`, `validation_end` или вложенного `metrics_ready`. RunManager выставляет finished и посылает run_end, но исходный обработчик после возврата продолжает работу.

Воспроизведено:

- Остановка во время `validation_end` → `run.stop=True`, но `run.phase='train'` и **`train_resume` после `run_end`**.
- Остановка во время `train_step_end` на шаге validation → `run.stop=True`, но `run.phase='validation'`, `validation_due=True` и **`validation_start` после `run_end`**.

Это напрямую мешает корректному EarlyStopping: ресурсы уже могут быть закрыты, а более поздние события продолжают поступать.

Исправление: после сигналов, допускающих изменение stop, повторно проверять состояние; делать завершение согласованным, сбрасывать pending validation; отдельно тестировать остановку из metrics_ready. Начать с RunManager, не переписывая весь event bus.

### E5. WeightProduct зависит от порядка модулей — P1

Место: `modules/control/weights.py`, `WeightProduct.reaction()` и `LinearWeightSchedule.reaction()`.

Оба слушают `train_step_end`. Когда Product расположен раньше Schedule, он читает старый вес.

Воспроизведено после первого step:

| Порядок модулей | base weight | effective weight |
|---|---:|---:|
| Schedule → Product | 2 | 2 |
| Product → Schedule | 2 | 1 |

Та же категория зависимости возникает с adaptive/frequency controllers на общих событиях. Это противоречит требованию не полагаться на порядок слушателей YAML.

Исправление: явно вызвать WeightProduct в pipeline после обновления источников либо посылать отдельное событие о готовности нужного набора весов. Если источников несколько, событие должно означать готовность всей нужной композиции.

### E6. MetricAdaptiveWeights реагирует на чужой набор метрик — P2

Место: `modules/control/weights.py`, строки 204–210.

Обработчик проверяет только event и phase, игнорируя `prefix`/source. И SegmentationMetrics, и SegmentationDetectionMetrics публикуют `metrics_ready(phase='validation')`.

Воспроизведение: событие с `prefix='detection.validation'` запускает чтение `metrics.validation.precision`. Если pixel metrics ещё не опубликованы, получается **KeyError**; если опубликованы раньше, возможны повторные EMA-обновления по устаревшим данным.

Исправление: явный фильтр источника/metric prefix либо отдельное именованное событие. Проверка наличия ключа сама по себе не решает повторное применение устаревших метрик.

### E7. Overlap на `[B,C]` падает — P2

Место: `ops/losses.py`, `_overlap_stats()` и sample aggregation.

Для prediction/target `[B,C]` spatial_dims пуст. `sum(())` в проверенном PyTorch сворачивает весь tensor, после чего reduction по dim=1 вызывает **IndexError**.

Исправление: нормализовать форму до `[B,C,1]` или явно обрабатывать отсутствие spatial dimensions. Если классификационный overlap намеренно не поддерживается, отклонять форму понятным ValueError до вычислений и документировать ограничение.

### E8. GeneralizedDice с полностью исключёнными классами и smooth=0 возвращает 1 — P2

Место: `ops/losses.py`, `generalized_dice_objective()`, строки 217–227.

Воспроизведено: пустые targets, multilabel, `empty_target='ignore'`, `smooth=0` → **loss=1** вместо нейтрального исключения. Градиент полезного обучения отсутствует. При положительном smoothing проблема численно скрыта почти нулевым остатком.

Исправление: явно обработать отсутствие валидной массы и вернуть связанный с графом ноль.

### E9. Вложенные константы изменяемы — P2

Место: `core/context.py`, строки 14–15 и 46–48.

`context.const('nested')['x']=9` меняет исходную константу, если nested — dict. Аналогично изменяемы списки. Это нарушает обещание неизменяемых констант.

Исправление: рекурсивно заморозить поддерживаемые конфигурационные контейнеры либо возвращать независимые копии. Это относится к конфигурации; копирование моделей при каждом чтении не требуется.

### E10. Нулевой запуск открывает progress после run_end — P2

Места: `RunManager._on_run_started()`, `Engine.signal()`, `TqdmReporter.reaction()`.

При `max_steps=0` и порядке Run → Progress вложенное `run_end` доставляется до того, как Progress получает исходное `run_started`. Затем Progress создаёт bar, который уже некому закрыть.

Воспроизведено с простым тестовым bar: после run_started, `closed=False`, хотя run завершён. Штатный тест проверяет только stop/phase, поэтому этого не замечает.

Исправление: защитить reporter от запуска после терминального состояния и отдельно зафиксировать семантику вложенных событий. Более общую очередь сигналов вводить только после оценки влияния на существующие event-порядки.

## Семантические расхождения и ограничения — не смешивать с багами

### S1. Positive/negative cost weights могут полностью сокращаться

`BCEWithLogits` и BinaryFocal включают cost weights в `element_weight`, а reduction делит на сумму этих же весов.

Для полностью пустой маски отрицательный коэффициент постоянен по пикселям:

`sum(w_negative * loss) / sum(w_negative) = mean(loss)`.

Проверено: negative_weight 0.1, 1 и 10 на пустой маске дают одинаковый BCE **0.693147**. На смешанной positive/negative маске отношение коэффициентов влияет на spatial weighting; утверждать, что эти веса всегда бесполезны, неверно.

Это соответствует выбранному normalized weighted mean, но не означает «в десять раз сильнее штрафовать пустую маску». Для такого поведения нужны отдельные cost multipliers в числителе с независимой маской/нормировкой. Следует развести понятия mask, importance-weighted mean и cost factor, не менять формулу молча.

### S2. Case/class weights задают относительную важность внутри sample

При `aggregation='sample'` сначала берётся нормированное среднее по классам, потом по sample. Если C=1, ненулевой case_class_weight сокращается. Поэтому PresenceWeights сам по себе не регулирует соотношение present/empty примеров одно-канальной сегментации.

Проверено: веса пустого случая 0.01, 1 и 100 дали одинаковый loss **0.659091**.

При `empty_target='ignore'` и единственном присутствующем классе в каждом sample также сокращается class_weight этого класса. В проверке замена `[1,1]` на `[1,100]` практически не изменила loss.

Это не ошибка реализации относительно документа с формулой двух последовательных средних. Но обещание произвольного управления «важностью редкого положительного случая» требует уточнения. Нужен явный выбор между:

- равной важностью sample с относительными class weights внутри каждого sample;
- общим weighted mean по валидным `[B,C]` парам;
- отдельным sample_weight, отражающим нужную важность случая.

Batch aggregation TP/FP/FN — ещё одна математическая операция; не следует использовать её как незаметную замену среднего по sample-class losses.

### S3. Context проверяет доступ к ключам, а не внутренним объектам

Reader без `updates` может выполнить `ctx['x']['value']=9` или in-place изменение tensor. Проверено на dict. Полученный в setup model-reference также можно изменять без нового обращения к Context.

Это уже используется встроенными модулями: Forward меняет train/eval mode; optimizer меняет parameters, хотя demo декларирует только reads модели.

Практический путь: честно описать уровень гарантий, объявлять мутирующие зависимости явно и добавить проверки типичных ошибок. Не строить сложную proxy-систему для всех PyTorch-объектов и не копировать модель при чтении. Рекурсивная неизменяемость конфигурационных констант — отдельная, гораздо более простая задача.

### S4. Не вся собственная математика вынесена в ops

Focal, binary cost composition, foreground projection и detection-statistics содержатся непосредственно в Module-классах. В `GradientMonitor` и OptimizationManager дублируется norm calculation.

Вызов готового PyTorch loss из тонкого адаптера допустим. Но собственные формулы weighting/focal/detection лучше выделить в чистые функции и тестировать без Context. Особенно это важно для фиксации семантики S1–S2.

### S5. OptimizationManager не проверяет связь loss с optimizable parameters

На независимом `torch.tensor(1., requires_grad=True)` backward проходит, norm=0 и `did_step=True`, хотя параметров с gradients нет. Это воспроизведено. Формально optimizer.step вызван, но обещание «реального update» двусмысленно.

Нужно отличать нормальный нулевой gradient от полного отсутствия gradients. Полезен диагностический режим, ловящий detached/wrong loss. При передаче готового optimizer norm/clipping желательно считать по его param_groups, а не безусловно по всем parameters model.

## Какие задачи ещё неудобно покрывать

«Возможно написать Module» не равно «задача удобно поддерживается готовыми компонентами». Ниже оценивается именно текущий набор библиотеки.

| Задача | Что уже есть | Конкретный пробел | Минимальное расширение |
|---|---|---|---|
| Binary/multilabel classification | Forward, BCE, focal | Потеря weights на `[B,C]`; overlap на этой форме падает | Исправить reduction/формы, recipe и соответствующие metrics |
| Scalar/tabular regression | MSE/L1, Forward | Выход `[B]` вызывает ValueError; нет regression metrics | Поддержка `[B]`/`[B,1]`, MAE/RMSE и dataset recipe |
| Semantic segmentation с void/unlabeled pixels | CE, overlap, metrics | Ignore-index есть только в CE/focal; Dice one_hot и multiclass bincount падают на -100 | Единый valid-mask/ignore контракт для loss и metrics |
| 3D medical segmentation | Размерно-независимые overlap losses, MONAI wrapper | Device/data путь и полноценная sliding-window validation не собраны | Eval/no-grad/autocast policy, volumes/spacing, recipe |
| NLP, causal LM, VLM/Florence | Tokenizer/processor/model/forward/generate | Нет готового collate с labels/attention mask, token shift/layout, sequence metrics | Task data adapter, label preparation, recipe; можно использовать штатный model.loss через HFForward |
| Diffusion training | Noise/timesteps/add_noise/model forward, MSE | Нет полного пути загрузки train-компонентов, latents/conditioning/target preparation и train/eval precision policy | Отдельные loaders/target modules и epsilon-prediction recipe; другие objectives отдельно |
| GAN, adversarial, несколько optimizer | Несколько OptimizationManager и явный DSL | Нет удобных detach/freeze/unfreeze и управления backward отдельно от step | Малые gradient-control модули, проверенный двух-optimizer recipe |
| Detection, instance/panoptic segmentation | Generic Forward, presence-метрики по маске | Нет variable-size targets, matching, box/mask metrics, object weights | Task-specific ops/adapters; нынешний SegmentationDetectionMetrics не является object-level mAP |
| Contrastive learning, distillation, teacher–student | Multi-input/output Forward, LossMixer | Нет pair sampling, contrastive/KL objectives, EMA teacher/queue | Независимые ops, sampling и teacher-update modules |
| RL / online interaction | while/when/iterate, общий Context | Нет env interaction, rollout/replay, returns/advantages, actor/critic recipes | Отдельный набор domain modules; пока не позиционировать как готовую поддержку |
| Distributed / крупные модели | Общая композиция модулей | Нет DDP/FSDP, samplers, metric all-reduce, rank-aware I/O, coordinated stop | Узкая интеграция и отдельные recipes; большая работа без ML-логики в core |
| Долгое воспроизводимое обучение | Save model/optimizer/step | Нет загрузки и полного восстановления run/controllers/scaler/RNG/data state | LoadCheckpoint и небольшой state_dict protocol для stateful modules |

### Уточнения к таблице

**Regression.** Воспроизведено: MSE на prediction/target `[3]` вызывает `Pointwise loss must have shape [B, C, ...]`. Приведение к `[B,1]` работает как обход; это ограничение удобства, а не неверная математика на документированной форме.

**Void labels.** На target с -100 Dice падает с `Class values must be non-negative`, multiclass metrics — с ошибкой bincount. Параметра ignore_index у них нет. Просто занулить element weights недостаточно: one_hot происходит раньше.

**MONAI.** Адаптер `MonaiSlidingWindowInferer` сам не задаёт eval/no_grad. С подменённым inferer и обычной обучаемой моделью проверено: `training=True`, `grad_enabled=True`, результат требует gradient. Это риск текущей границы интеграции, не end-to-end тест настоящего MONAI. Нужны явные режимы либо документированная отдельная module-композиция управления режимами.

**NLP.** Общий CE ожидает classes на оси 1; прямое подключение logits `[B,T,V]` требует перестановки осей или другого адаптера. HFForward позволяет извлечь готовый `loss` модели, если правильно подготовлен input batch с labels; поэтому нельзя утверждать, что language-model training невозможно вообще.

**Accumulation.** Текущая реализация делит loss каждого micro-batch на постоянный accumulation. Это подходит для равных по смыслу micro-batch means. При разном числе валидных tokens/pixels не эквивалентно среднему по всем tokens/pixels. Нет flush незавершённой accumulation-группы при окончании конечного источника; политика drop/flush/error должна быть явной.

**Resume.** Недостаточно загрузить веса модели и optimizer: RunManager.setup обнуляет время, schedules зависят от шага, адаптивные веса и scaler имеют состояние. Exact resume посреди accumulation дополнительно требует gradients или ограничения сохранения границей update.

**Данные и device.** DataLoaderSource адаптирует уже созданный loader; готового универсального dataset/loader construction и рекурсивного batch.to(device) нет. Для реальных experiments понадобится небольшой producer loader и перенос tensor leaves в dict/list/tuple. Это не требует глобального device manager в Engine.

**Метрики и завершение.** Train metrics в demo публикуются только на validation_start. При отключённой final validation хвост train-окна может остаться неопубликованным. Engine также не предоставляет exception/cleanup lifecycle; ресурсы loggers гарантированно закрываются только при штатном run_end. Это стоит решить отдельно от политики времени обучения.

## Рекомендуемый порядок работ

1. **Корректность вычислений.** Исправить E1–E3, E7–E8; добавить минимальные математические tests с half dtype, ignored samples, `[B]`, `[B,C]`, нулевой маской и backward. Зафиксировать mask/cost/normalization semantics S1–S2 до изменения весов.
2. **Корректность событий.** Исправить E4–E6 и E10. Проверять одинаковый итог при допустимой перестановке независимых слушателей; терминальное состояние должно оставаться терминальным.
3. **Уточнить архитектурные обещания.** Разделить mutation объекта и замену ключа; заморозить constants; вынести собственную математику из modules. Свести две версии философии к одной актуальной спецификации.
4. **Минимум для реальных запусков.** BatchToDevice/ModelToDevice, явные model/grad modes, state save/load, простой LR scheduler и EarlyStopping после исправления lifecycle. Не добавлять общий Trainer.
5. **Проверить обобщаемость recipes.** Сначала реальные segmentation и scalar regression; затем GAN с двумя optimizer; затем causal LM или diffusion training. Каждый recipe должен обнаруживать конкретные недостающие примитивы, а не приводить к новому task type внутри Engine.

Большинство изменений относится к `ops`, `modules`, `integrations` и recipes. Для core необходимы прежде всего честный контракт Context и определённая семантика сигналов; оснований превращать его в ML-specific orchestration layer аудит не выявил.

## Как усилить тесты без разрастания suite

Нынешние 23 теста проверяют полезный happy path, но почти не проверяют реальное поведение basic loss-модулей, reentrant stop и порядок consumers/producers.

Обязательные регрессии после исправлений:

- FP16 weighted reduction: большие суммы и полностью нулевой вес, finite forward/backward.
- CE/focal: добавление ignored sample не меняет loss валидной части.
- BCE: `[B,C]` element mask исключает неизвестный label; cost semantics проверены отдельно.
- Overlap: `[B,C]`, all-ignored GeneralizedDice со smooth=0, void labels с явной политикой.
- RunManager: stop из train_step_end и metrics_ready; ни resume, ни validation после терминального завершения.
- WeightProduct: готовый effective weight после обновления всех источников, независимо от положения listeners.
- MetricAdaptiveWeights: событие другой группы метрик не вызывает обновления.
- Constants: вложенные list/dict нельзя незаметно изменить.
- Один настоящий optional integration recipe в отдельной среде; mocks не считать end-to-end проверкой.
