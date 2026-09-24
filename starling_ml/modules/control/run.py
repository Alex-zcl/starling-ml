"""Step-driven управление жизненным циклом эксперимента."""

from ...core.module import Module


class RunManager(Module):
    """Ведёт global step и решает, когда нужна валидация или остановка.

    Менеджер не знает о модели, метриках и optimizer. Он меняет только общее
    состояние запуска и посылает события, на которые независимо реагируют
    остальные модули.
    """

    def setup(
        self,
        max_steps,
        validate_every=0,
        validate_at_start=False,
        validate_at_end=True,
        step="run.step",
        phase="run.phase",
        validation_index="run.validation_index",
        validation_due="run.validation_due",
        stop="run.stop",
    ):
        self.max_steps = int(max_steps)
        self.validate_every = int(validate_every or 0)
        self.validate_at_start = bool(validate_at_start)
        self.validate_at_end = bool(validate_at_end)
        self.step_key = step
        self.phase_key = phase
        self.validation_index_key = validation_index
        self.validation_due_key = validation_due
        self.stop_key = stop

        self.context[step] = 0
        self.context[phase] = "train"
        self.context[validation_index] = 0
        self.context[validation_due] = False
        self.context[stop] = self.max_steps <= 0

    def reaction(self, signal, source=None, **payload):
        if signal == "run_started":
            self._on_run_started()
        elif signal == "step_completed":
            self._on_step_completed()
        elif signal == "validation_completed":
            self._on_validation_completed()
        elif signal == "stop_requested":
            self._finish_run()

    def _on_run_started(self):
        """При необходимости запускает validation до первого train-step."""
        if self.context[self.stop_key]:
            # Нулевой лимит является уже завершённым запуском. Фазу меняем до
            # сигнала, чтобы все слушатели увидели согласованное состояние.
            self.context[self.phase_key] = "finished"
            self.signal("run_end", step=self.context[self.step_key])
        elif self.validate_at_start:
            self._start_validation()

    def _on_step_completed(self):
        """Увеличивает только число завершённых логических train-step."""
        if self.context[self.stop_key]:
            return

        step = self.context[self.step_key] + 1
        self.context[self.step_key] = step
        self.signal("train_step_end", step=step)
        if self.context[self.stop_key]:
            return

        periodic = self.validate_every > 0 and step % self.validate_every == 0
        final_validation = step >= self.max_steps and self.validate_at_end

        if periodic or final_validation:
            self._start_validation()
        elif step >= self.max_steps:
            self._finish_run()

    def _start_validation(self):
        """Переключает фазу до события, чтобы слушатели видели новое состояние."""
        if self.context[self.stop_key] or self.context[self.validation_due_key]:
            return
        self.context[self.phase_key] = "validation"
        self.context[self.validation_due_key] = True
        self.signal("validation_start", step=self.context[self.step_key])

    def _on_validation_completed(self):
        """Сначала завершает validation-события, затем возвращает train или stop."""
        if not self.context[self.validation_due_key]:
            return

        index = self.context[self.validation_index_key] + 1
        self.context[self.validation_index_key] = index
        self.signal("validation_end", step=self.context[self.step_key], validation_index=index)
        self.context[self.validation_due_key] = False
        if self.context[self.stop_key]:
            return

        if self.context[self.step_key] >= self.max_steps:
            self._finish_run()
        else:
            self.context[self.phase_key] = "train"
            self.signal("train_resume", step=self.context[self.step_key])

    def _finish_run(self):
        """Идемпотентно отмечает остановку и посылает финальное событие."""
        if self.context[self.stop_key]:
            return
        self.context[self.stop_key] = True
        self.context[self.validation_due_key] = False
        self.context[self.phase_key] = "finished"
        self.signal("run_end", step=self.context[self.step_key])

    def state_dict(self):
        return {key:self.context[key] for key in (self.step_key,self.phase_key,self.validation_index_key,self.validation_due_key,self.stop_key)}

    def load_state_dict(self,state):
        for key,value in state.items():self.context[key]=value
        # Увеличенный max_steps позволяет продолжить завершённый checkpoint.
        if self.context[self.step_key] < self.max_steps:
            self.context[self.stop_key]=False
            if not self.context[self.validation_due_key]:self.context[self.phase_key]="train"
