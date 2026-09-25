"""Глобальный lifecycle запуска без знания train/validation/test."""

from ...core.module import Module


class RunManager(Module):
    """Считает успешные optimizer steps и завершает весь эксперимент.

    ``defer_finish=True`` включает новый режим 0.4: финальные фазы выполняет
    PhaseManager, после чего посылает ``phases_completed``. Без этого флага
    сохраняется lifecycle 0.3 для старых YAML-конфигов.
    """

    def setup(
        self,
        max_steps,
        defer_finish=False,
        step="run.step",
        status="run.status",
        finish_requested="run.finish_requested",
        stop="run.stop",
        validate_every=0,
        validate_at_start=False,
        validate_at_end=True,
        phase="run.phase",
        validation_index="run.validation_index",
        validation_due="run.validation_due",
    ):
        self.max_steps = int(max_steps)
        self.defer_finish = bool(defer_finish)
        self.step_key, self.stop_key = step, stop

        if self.defer_finish:
            self.status_key = status
            self.finish_requested_key = finish_requested
            self.context[step] = 0
            self.context[status] = "created"
            self.context[finish_requested] = self.max_steps <= 0
            self.context[stop] = False
            return

        # Совместимость с 0.3: старый RunManager продолжает исполнять validation.
        self.validate_every = int(validate_every or 0)
        self.validate_at_start = bool(validate_at_start)
        self.validate_at_end = bool(validate_at_end)
        self.phase_key = phase
        self.validation_index_key = validation_index
        self.validation_due_key = validation_due
        self.context[step] = 0
        self.context[phase] = "train"
        self.context[validation_index] = 0
        self.context[validation_due] = False
        self.context[stop] = self.max_steps <= 0

    def reaction(self, signal, source=None, **payload):
        if self.defer_finish:
            self._reaction_v04(signal)
        else:
            self._reaction_v03(signal)

    def _reaction_v04(self, signal):
        if signal == "run_started":
            if self.context[self.stop_key]:
                return
            self.context[self.status_key] = (
                "finishing" if self.context[self.finish_requested_key] else "running"
            )
            if self.context[self.finish_requested_key]:
                self.signal("run_finish_requested", step=self.context[self.step_key])
        elif signal == "step_completed":
            if self.context[self.stop_key] or self.context[self.finish_requested_key]:
                return
            step = self.context[self.step_key] + 1
            self.context[self.step_key] = step
            if step >= self.max_steps:
                self.context[self.finish_requested_key] = True
                self.context[self.status_key] = "finishing"
            self.signal(
                "run_step_end",
                step=step,
                finish_requested=self.context[self.finish_requested_key],
            )
        elif signal == "stop_requested":
            if self.context[self.stop_key] or self.context[self.finish_requested_key]:
                return
            self.context[self.finish_requested_key] = True
            self.context[self.status_key] = "finishing"
            self.signal("run_finish_requested", step=self.context[self.step_key])
        elif signal == "phases_completed":
            self._finish_v04()

    def _finish_v04(self):
        if self.context[self.stop_key]:
            return
        self.context[self.stop_key] = True
        self.context[self.status_key] = "finished"
        self.signal("run_end", step=self.context[self.step_key])

    # Ниже изолирован совместимый lifecycle 0.3. Новые recipes его не используют.
    def _reaction_v03(self, signal):
        if signal == "run_started":
            if self.context[self.stop_key]:
                self.context[self.phase_key] = "finished"
                self.signal("run_end", step=self.context[self.step_key])
            elif self.validate_at_start:
                self._start_validation_v03()
        elif signal == "step_completed":
            self._step_v03()
        elif signal == "validation_completed":
            self._validation_completed_v03()
        elif signal == "stop_requested":
            self._finish_v03()

    def _step_v03(self):
        if self.context[self.stop_key]:
            return
        step = self.context[self.step_key] + 1
        self.context[self.step_key] = step
        self.signal("train_step_end", step=step)
        periodic = self.validate_every > 0 and step % self.validate_every == 0
        final = step >= self.max_steps and self.validate_at_end
        if periodic or final:
            self._start_validation_v03()
        elif step >= self.max_steps:
            self._finish_v03()

    def _start_validation_v03(self):
        if self.context[self.stop_key] or self.context[self.validation_due_key]:
            return
        self.context[self.phase_key] = "validation"
        self.context[self.validation_due_key] = True
        self.signal("validation_start", step=self.context[self.step_key])

    def _validation_completed_v03(self):
        if not self.context[self.validation_due_key]:
            return
        index = self.context[self.validation_index_key] + 1
        self.context[self.validation_index_key] = index
        self.signal("validation_end", step=self.context[self.step_key], validation_index=index)
        self.context[self.validation_due_key] = False
        if self.context[self.step_key] >= self.max_steps:
            self._finish_v03()
        else:
            self.context[self.phase_key] = "train"
            self.signal("train_resume", step=self.context[self.step_key])

    def _finish_v03(self):
        if self.context[self.stop_key]:
            return
        self.context[self.stop_key] = True
        self.context[self.validation_due_key] = False
        self.context[self.phase_key] = "finished"
        self.signal("run_end", step=self.context[self.step_key])

    def state_dict(self):
        if self.defer_finish:
            keys = (self.step_key, self.status_key, self.finish_requested_key, self.stop_key)
        else:
            keys = (
                self.step_key,
                self.phase_key,
                self.validation_index_key,
                self.validation_due_key,
                self.stop_key,
            )
        return {key: self.context[key] for key in keys}

    def load_state_dict(self, state):
        for key, value in state.items():
            self.context[key] = value
        if self.context[self.step_key] < self.max_steps:
            self.context[self.stop_key] = False
            if self.defer_finish:
                self.context[self.finish_requested_key] = False
                self.context[self.status_key] = "running"
            elif not self.context[self.validation_due_key]:
                self.context[self.phase_key] = "train"
