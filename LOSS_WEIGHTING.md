# Весá и loss в 0.3.0: фактический контракт

## Три смысла коэффициента

Mask отвечает, известна ли разметка; ложные элементы полностью исключаются.
Importance меняет относительный вклад и входит в числитель и знаменатель.
Cost умножает ошибку и сам по себе не меняет знаменатель.
Статистика или ручной приоритет — источник коэффициента, а не место его применения.

| Уровень | Межклассовый коэффициент | Positive | Negative |
|---|---|---|---|
| Элемент | CE/focal label_cost или TargetClassWeights map | BCE/focal positive_cost; внутри overlap positive_weight | BCE/focal negative_cost; внутри overlap negative_weight |
| Готовая ошибка класса | class_weight | present_weight | empty_weight |

Другие роли: valid_mask, element_weight, case_class_weight `[B,C]`, sample_weight `[B]`,
empty_cost, term weight. Несколько ролей можно применять совместно, но каждый источник
должен быть явным: частотная поправка, ручная важность, schedule или adaptive factor.

## Источник: statistics датамодуля

`DatasetStats` обходит полный dataset отдельно от training iterator и публикует:

- positive/negative/valid element counts по классам;
- positive/negative/unknown presence counts исследований и samples;
- study_count, sample_count, total_element_count;
- class_names, task_mode, split, dataset_fingerprint, transform_scope,
  sample_unit, completeness.

Для multiclass counts считаются one-vs-rest. Для multilabel unknown канала не является
negative. Counts — целые события; soft targets не превращаются скрыто в counts.
`total_element_count` — число пространственных элементов без множителя числа каналов.

Один study_id дедуплицируется. **Для patches укажите sample_unit="patch"**: отсутствие
класса в наблюдаемых patches не доказывает отсутствие в исследовании. Отрицательное
study presence допустимо при записи, явно обозначенной `study_complete=True`, с полной
валидной разметкой или явным `presence_known`. Положительное наблюдение доказывает
присутствие и без полной разметки. Объединение пересекающихся patches не удаляет
повторные pixels: element counts отражают переданный dataset. Для source-level counts
подавайте исходные исследования, не список перекрывающихся patches.

`DatasetWeights(level="element"|"case", balance="class"|"binary")` выбирает counts.
Для case уровень `unit="study"|"sample"` задаётся явно. Используется
`(q/f)**gamma`, где f — частота, q — желаемая доля. gamma=0 — без частотной
компенсации, .5 — частичная, 1 — полная. Normalization: `expectation`, `mean`, `none`.
Можно задать bounds и missing policy: `error`, `zero`, `neutral`. Отсутствующему классу
не назначается гигантский вес через epsilon. Legacy `FrequencyWeights` сохранён
для совместимости; анализатор рекомендует новый контроллер.

Пример: 20 positive и 80 empty studies при gamma=1, q=(.5,.5) дают
present=2.5, empty=.625. Dataset-масса каждой группы равна 50.
Если class A занимает 1% pixels, но встречается в 20% studies, это две разные частоты.
Oversampling/foreground crops меняют обучающее распределение и требуют повторного
обдумывания коэффициентов. Metadata `sampling_changed=True` включает предупреждение.

## BCE и binary focal

Для soft target y в [0,1], logit z, p=sigmoid(z):

```
loss = positive_cost * y * (1-p)**gamma * softplus(-z)
     + negative_cost * (1-y) * p**gamma * softplus(z)
```

gamma=0 даёт BCE. Positive/negative costs стоят **только в числителе**.
При двух элементах y=[1,0], z=[0,0] и costs=[3,1] среднее равно 2*log(2),
а не log(2). Применять для разной цены FN/FP или pixel imbalance.
Для binary `[B]` имеется один канал: vector class cost должен иметь одну запись.
Sample priorities задаются через sample_weight, не через channel cost.

## Softmax CE и multiclass focal

Для hard target k: `loss_i = -label_cost[k]*(1-p[i,k])**gamma*log(p[i,k])`.
Вероятность берётся непосредственно из softmax; `exp(-weighted_CE)` не используется.
Для soft labels и label smoothing используется сумма компонентных потерь по классам.
Channel-last logits поддерживаются через class_dim=-1.

Пример `[background,A,B]`, label_cost=[.2,1,3]: ошибка фонового элемента стоит .2,
ошибка B — 3. Эти costs не добавляются в denominator автоматически.
Если нужно weighted mean по target-class importance, постройте `TargetClassWeights`
и передайте результат как element_weight с element_normalization="weight_sum".
При label smoothing это и per-class компонентные costs — разные objectives.

Background index задаётся явно; 0 не исключается автоматически.
Без фонового канала каждый валидный элемент должен принадлежать одному из моделируемых
классов. Иначе используйте ROI/ignore или добавьте background. Softmax без background
не приобретает способность предсказывать «ни один класс».

## Pointwise reducers

`element_normalization="weight_sum"`: N=sum(mask*element_weight*loss),
D=sum(mask*element_weight). `"valid_count"`: D=sum(mask); element_weight действует
как cost. Mask интерпретируется логически; дробную importance передавайте отдельно.

CE/focal default `reduction="element_mean"` объединяет элементы; sample_weight
умножает N и D соответствующих samples. `sample_mean` сначала нормирует каждый sample,
затем усредняет валидные samples с sample_weight.

BCE/regression default `pair_mean`: сначала spatial mean каждого `[B,C]`, затем
внешний reducer. Доступен `element_mean` для общего среднего по всем валидным элементам.
Он отличается от pair_mean при разном количестве валидных pixels в масках.
При `[B]`/`[B,C]` element importance сохраняется на внешнем уровне; вес не исчезает
из-за отсутствия spatial axes. Нулевая cost с valid_count оставляет case в знаменателе.

## Overlap: внутренние pixel weights

Dice/IoU/Tversky принимают независимые element_weight и positive_weight/negative_weight:

```
w_i = valid_i * element_weight_i * (positive_weight_c*y_i + negative_weight_c*(1-y_i))
TP = sum(w*p*y); FP = sum(w*p*(1-y)); FN = sum(w*(1-p)*y)
DiceLoss = 1 - (2*TP + smooth)/(2*TP + FP + FN + smooth)
IoULoss  = 1 - (TP + smooth)/(TP + FP + FN + smooth)
TverskyLoss = 1 - (TP + smooth)/(TP + alpha*FP + beta*FN + smooth)
```

Positive weight меняет TP и FN; negative — FP. Эти weights отличаются от present/empty.
Constant multiplier всех TP/FP/FN канала сокращается при smooth=0 и меняет относительную
силу smoothing при smooth>0. Чтобы взвесить элементы по их истинному классу в softmax,
используйте TargetClassWeights map, например `[B,H,W]`, а не broadcast `[C]`.
Spatial map `[B,...]` расширяется вдоль каналов; полная `[B,C,...]` тоже допустима.

## Overlap: внешние weights и пустые маски

Сначала вычисляется loss_bc, затем:
`weight_bc = class_weight_c * case_class_weight_bc * sample_weight_b * presence_factor_bc`.
Presence определяется на валидном target до pixel costs. `present_weight` применяется
при присутствии класса, `empty_weight` — при пустой target mask.
Полностью неизвестные пары не участвуют.

| Reducer | Формула и смысл |
|---|---|
| pair_mean (default) | sum(weight*loss)/sum(weight); равноправные sample-class пары до назначения weights |
| sample_mean | сначала class mean каждого sample; затем sample-weighted mean; samples имеют равный исходный приоритет |
| class_macro | сначала case mean каждого класса; затем class-weighted mean активных классов |
| aggregation=batch_stats | сначала объединить TP/FP/FN, затем один overlap на канал; это другой objective |

Для class_macro class_weight стоит только во внешнем mean, case/sample weights —
внутри class mean. Если 90 случаев A имеют loss .2 и 10 B — .8, pair mean=.26,
class_macro=.5. Если каждый класс имеет одинаковое число валидных пар, частотный
class_weight усиливает редкий класс вместе с его пустыми случаями; это не просто
выравнивание числа слагаемых.

В batch_stats sample/case weights входят **внутрь TP/FP/FN**, present/empty выбираются
для глобально объединённой маски. reduction не меняет batch_stats на sample mean.
В режиме sample smoothing применяется отдельно к каждому case, в batch_stats —
один раз к объединённой статистике.

Empty policies:

- ignore: исключить пустые target pairs;
- standard (alias penalize): обычная формула со smoothing;
- false_positive: средняя взвешенная predicted foreground mass на пустой mask.

`empty_cost` умножает уже вычисленную ошибку пустой маски и не входит в denominator.
При полностью пустом batch empty_weight=.1 сокращается, empty_cost=.1 уменьшает loss
в 10 раз. При empty_target=ignore оба не действуют. Для уменьшения агрессивности на
пустых масках используйте false_positive и подходящий empty_cost.

`include_classes=[1,2]` исключает только внешние class terms Dice. Background pixels
остаются negative для этих каналов; CE/focal сохраняет background term независимо.

## Generalized Dice

Реализован общий взвешенный ratio, **без автоматического inverse-square volume**:
`1 - (2*sum_c(a_c*TP_c)+smooth)/(sum_c(a_c*(P_c+Y_c))+smooth)`.
Element/case/sample weights входят внутрь stats; a_c — class_weight активных классов.
При всех исключённых классах результат — differentiable zero. Если нужны традиционные
inverse-volume факторы, задайте их явно. Эта операция не эквивалентна mean Dice.

## Композиции

1. Невзвешенный Dice + focal с pixel costs: разные modules, свои параметры, затем Mixer.
2. Weighted overlap: element map и/или positive/negative weights только внутри TP/FP/FN.
3. Present/empty weighting: внешнее распределение внимания между непустыми/пустыми cases.
4. Class priorities: class_weight либо class_macro; ручная важность независима от частоты.
5. Все уровни вместе: допустимо при осмысленных источниках; повторную компенсацию
   должен проверить пользователь по сводке анализатора.

LossMixer принимает готовые scalar losses: `sum` или `normalized_sum` (деление на сумму
модулей term weights). Общий scale в normalized_sum сокращается. Native Objective
metadata сохраняется. Внешний scalar внутри Mixer не допускает exact accumulation/DDP
без достаточных stats; безопасно завершится ошибкой вместо скрытого micro mean.

## Accumulation и DDP

Objective содержит суммируемые stats и функцию финальной редукции. Для среднего
объединяются N/D, для class_macro — векторы N_c/D_c, для batch overlap — TP/FP/FN.
Проверки сравнивают и loss, и gradients с единым batch. Exact mode сохраняет autograd
графы microbatches, поэтому **не обещает экономию памяти**. BatchNorm, stochastic
layers и contrastive negatives могут менять сам forward при разбиении batch;
эквивалентность редукции не устраняет это отличие.

DDP использует differentiable sum stats и DDP gradient averaging. Все ranks должны
проходить одинаковые collectives и одинаковое число optimizer steps. Ragged validation
метрики объединяются при завершении общего окна. Полный distributed запуск не проверен
из-за запрета сокетов в среде. GPU AMP также требует отдельного прогона.
