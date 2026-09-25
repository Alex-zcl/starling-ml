> Применяется к общему step-driven pipeline. В 0.4.0 RunManager отвечает за global step/stop, а PhaseManager — за фазовые переходы; актуальные ограничения описаны в README.md и RESUME_NOTES.md.

# Step-driven lifecycle

## Состояние запуска

`RunManager` создаёт global state:

```text
run.step
run.status
run.finish_requested
run.stop
```

`PhaseManager` создаёт `phase.name`, `phase.index`, `phase.step`, `phase.cycle` и
`phase.due`. Ключи `run.phase`, `run.validation_index`, `run.validation_due`
публикуются как compatibility view для конфигов 0.3.

`run.step` увеличивается только после сигнала:

```text
step_completed
```

Pipeline, а не optimizer и не Engine, определяет момент завершения логического step.

## События

```text
run_started
step_completed          внутренний запрос RunManager
run_step_end            публичное phase-neutral событие после увеличения run.step
phase_started
phase_step_end
phase_completed         внутренний запрос PhaseManager
phase_ended
phases_completed        подтверждение RunManager, что финальные фазы завершены
stop_requested
run_finish_requested
run_end
```

События `train_step_end`, `validation_start/end` и `train_resume` временно
сохраняются для совместимости существующих reporters.

## Базовый pipeline

```yaml
- signal: run_started

- while:
    condition: $ctx:run.stop
    equals: false
    body:
      - call: TrainData
      - call: TrainForward
      - call: Loss
      - call: Optimization

      - when:
          condition: $ctx:optim.main.did_step
          equals: true
          body:
            - signal: step_completed

      - when:
          condition: $ctx:phase.due
          equals: true
          body:
            - loop:
                count: $const:validation_steps
                body:
                  - call: ValidationData
                  - call: ValidationForward
                  - call: ValidationMetrics
                  - signal:
                      name: phase_batch_completed
                      payload: {phase: validation}
            - signal:
                name: phase_completed
                payload: {phase: validation}
```

## Gradient accumulation

При `accumulation: 4` модуль оптимизации четыре раза выполняет backward и только на четвёртом вызове устанавливает:

```text
optim.main.did_step = true
```

Следовательно:

```text
micro-batch        1 2 3 4 5 6 7 8
run.step           0 0 0 1 1 1 1 2
```

Validation, schedule и checkpoint могут использовать `run.step`, не зная размер accumulation.

## Конечный DataLoader

Для полного прохода по validation loader используется стандартный `StopIteration`:

```yaml
- iterate:
    source: ValidationData
    body:
      - call: ValidationForward
      - call: ValidationMetrics
      - signal: validation_step_end
```

`DataLoaderSource(cycle=false)` поднимает `StopIteration`, а Engine завершает блок `iterate`.

## Порядок метрик

На `phase_ended(phase=train)`:

```text
TrainMetrics -> publish -> metrics_ready(phase=train) -> reset
ValidationMetrics -> reset
```

На `phase_ended(phase=validation)`:

```text
ValidationMetrics -> publish -> metrics_ready(phase=validation) -> reset
```

Logger и weight managers слушают `metrics_ready`, а не надеются на порядок реакций на `validation_end`.

## Несколько optimizer-ов

Для GAN можно создать:

```text
GeneratorOptimization
DiscriminatorOptimization
```

Каждый публикует собственный `did_step`. Pipeline явно решает, когда их комбинация считается одним `step_completed`. В Engine специальной GAN-логики нет.
