# Философия движка — 0.4.0

## Пользователь работает с конфигом

Основной путь: импортировать `Engine`, получить `get_config(task)`, изменить обычный
словарь, показать `analyze_config(config)` и выполнить `Engine(config).run()`.
Для реальной задачи пользователь заменяет dataset/model factories и ключи batches.
Внутренние модули доступны для расширения, но их не нужно вручную собирать Python-кодом
при каждом эксперименте. Стандартные конфиги не разделяют изменяемые объекты между вызовами.

## Маленькое ядро, явный pipeline

Engine знает шесть общих операций: call, signal, loop, while, when, iterate.
Он не знает Dice, diffusion, optimizer, эпох или алгоритмов RL. Эти понятия принадлежат
модулям. Setup сортируется по ссылкам $ctx, а runtime порядок записан в конфиге.
Анализатор объясняет конфигурацию, но не переставляет её шаги и не меняет objective.
Новый алгоритм сначала реализуется простым Module или pure function; абстракция появляется
только при реальном повторении структуры.

## Context — явное владение

У каждого key один creator. Reads разрешает чтение, updates — замену значения,
mutates — изменение объекта по ссылке. Константы рекурсивно заморожены в обычных
конфигурационных контейнерах. Torch modules/tensors не оборачиваются в сложные proxies.
Это договор для расширений, а не защита от произвольного Python-кода: reads не делает
полученный torch object физически immutable. Встроенные configs объявляют мутации модели,
в том числе переключение train/eval и optimizer updates.

## Математика живёт в ops

Модуль получает tensors по именам, вызывает чистую функцию и публикует output.
Loss formula не читает Context, не меняет optimizer и не выбирает источник статистики.
Датамодуль считает counts; controller делает coefficients; loss применяет их;
LossMixer смешивает уже готовые scalar objectives.

Mask, importance и cost имеют разный смысл. Pixel frequency не равна study frequency.
Внутренний weight TP/FP/FN не равен множителю готового Dice. Positive/negative pixels
не равны present/empty masks. Эти различия должны быть видны в именах, документации,
знаменателях и сводке конфигурации. Автоматическая inverse frequency на всех уровнях
запрещена как скрытый выбор: пользователь задаёт источники и композицию явно.

## Reduction — часть objective

pair_mean, sample_mean, class_macro и batch_stats не являются взаимозаменяемыми
способами «сделать scalar». Они задают разные задачи оптимизации. Exact accumulation
объединяет достаточные stats, а не локальные средние. Price точности — удержание graphs;
это явно описанная реализация, не обещание экономии GPU памяти.

FP16/BF16 sums выполняются в FP32. Нулевой denominator даёт differentiable zero,
не скрытый NaN. Полностью невалидные samples/classes исключаются из соответствующих
средних. Неизвестная разметка не превращается в отрицательную.

## Lifecycle, фазы и state

RunManager ведёт global logical step, status и окончание всего запуска, но не знает имён
train/validation/test. OptimizationManager сообщает did_step; pipeline решает, какой
optimizer завершает логический шаг при нескольких моделях. PhaseManager отвечает за
активную фазу, переходы, циклы и обязательную финальную фазу до run_end.

Фаза не является скрытым глобальным режимом для всех модулей. Stateful metrics и losses
получают отдельные экземпляры на фазу; shared model/optimizer остаются run-scoped.
`phase_module` уменьшает дублирование деклараций, но после композиции Engine видит обычные
явные modules и contracts. Validation использует явные grad/mode настройки. Stop
идемпотентен и не должен повторно запускать validation.

## Композиция и профили

Recipe, пользовательские modules и monitoring — независимые фрагменты. `compose` не
использует last-write-wins: совпадение имён является ошибкой, пока замена не объявлена
явно. Поэтому набор логгеров не размножает task-конфиги. Standard/tracking профили
добавляют наблюдение, а ClearML остаётся отдельным явным действием с внешним эффектом.

Каждый stateful module описывает state_dict/load_state_dict. Checkpoint-функции сохраняют
состояния и RNG вне Engine. Встроенный BatchSource сохраняет permutation/cursor;
произвольный DataLoader или environment не получает автоматически точный resume.
Снимок посреди accumulation запрещён: нужно явно завершить либо отбросить группу.
Cleanup вызывается при завершении и ошибках run; ошибки cleanup не маскируют исходную.

## Проверяемые границы

Наличие wrapper не означает поддержку полного production pipeline. В отчёте различаются:
математический тест, synthetic end-to-end recipe, mock optional integration, реальный
GPU/distributed/pretrained запуск. Данные текущего отчёта важнее старых сообщений чата.
Непроверенный DDP или GPU нельзя объявлять готовым по наличию кода.

Production-интеграция должна добавлять конкретный dataset, preprocessing, task metrics,
resource lifecycle и воспроизводимую проверку. Она не должна расширять Engine новыми
флагами задач, молча исправлять формулы или менять нормировку при упаковке в pip.
