"""Универсальный менеджер фаз поверх глобального RunManager."""

from ...core.module import Module


class PhaseManager(Module):
    """Управляет train/validation-переходами, не считая optimizer steps."""

    def setup(
        self,
        validate_every=0,
        validate_at_start=False,
        validate_at_end=True,
        train="train",
        validation="validation",
        run_step="run.step",
        finish_requested="run.finish_requested",
        run_stop="run.stop",
        name="phase.name",
        index="phase.index",
        step="phase.step",
        cycle="phase.cycle",
        due="phase.due",
        legacy_phase="run.phase",
        legacy_validation_index="run.validation_index",
        legacy_validation_due="run.validation_due",
    ):
        self.validate_every = int(validate_every or 0)
        self.validate_at_start = bool(validate_at_start)
        self.validate_at_end = bool(validate_at_end)
        self.train, self.validation = train, validation
        self.run_step_key = run_step
        self.finish_requested_key = finish_requested
        self.run_stop_key = run_stop
        self.name_key, self.index_key = name, index
        self.step_key, self.cycle_key, self.due_key = step, cycle, due
        self.legacy_phase_key = legacy_phase
        self.legacy_validation_index_key = legacy_validation_index
        self.legacy_validation_due_key = legacy_validation_due
        self.completed = False

        self.context[name] = train
        self.context[index] = 0
        self.context[step] = 0
        self.context[cycle] = 0
        self.context[due] = False
        self.context[legacy_phase] = train
        self.context[legacy_validation_index] = 0
        self.context[legacy_validation_due] = False

    def reaction(self, signal, source=None, **payload):
        # Checkpoint мог быть сохранён после завершённого короткого запуска, а
        # затем загружен в конфиг с большим max_steps. RunManager в этом случае
        # снова открывает run; PhaseManager должен сделать то же самое.
        if signal == "run_started" and self.completed:
            if not self.context[self.run_stop_key] and not self.context[self.finish_requested_key]:
                self.completed = False
                self.context[self.name_key] = self.train
                self.context[self.legacy_phase_key] = self.train
                self.context[self.due_key] = False
                self.context[self.legacy_validation_due_key] = False
                self.context[self.step_key] = 0
            else:
                return
        if self.completed:
            return
        if signal == "run_started":
            self._on_run_started()
        elif signal == "run_step_end":
            self._on_run_step_end(payload)
        elif signal in {"phase_completed", "validation_completed"}:
            phase = payload.get("phase", self.context[self.name_key])
            if phase == self.validation:
                self._finish_validation()
        elif signal == "run_finish_requested":
            self._on_finish_requested()
        elif signal == "phase_batch_completed":
            phase = payload.get("phase", self.context[self.name_key])
            if phase == self.context[self.name_key]:
                self.context[self.step_key] = self.context[self.step_key] + 1
                self.signal(
                    "phase_step_end",
                    phase=phase,
                    step=self.context[self.step_key],
                    cycle=self.context[self.cycle_key],
                )

    def _on_run_started(self):
        if self.context[self.run_stop_key]:
            return
        if self.context[self.finish_requested_key]:
            self._complete_all()
        elif self.validate_at_start:
            self._start_validation()
        else:
            self.signal("phase_started", phase=self.train, index=0, cycle=0)

    def _on_run_step_end(self, payload):
        if self.context[self.run_stop_key] or self.context[self.name_key] != self.train:
            return
        run_step = int(payload.get("step", self.context[self.run_step_key]))
        phase_step = self.context[self.step_key] + 1
        self.context[self.step_key] = phase_step
        self.signal(
            "phase_step_end",
            phase=self.train,
            step=phase_step,
            run_step=run_step,
            cycle=self.context[self.cycle_key],
        )
        self.signal("train_step_end", step=run_step, phase=self.train)

        finishing = bool(self.context[self.finish_requested_key])
        periodic = self.validate_every > 0 and run_step % self.validate_every == 0
        if periodic or (finishing and self.validate_at_end):
            self._start_validation()
        elif finishing:
            self.signal(
                "phase_ended",
                phase=self.train,
                step=self.context[self.step_key],
                cycle=self.context[self.cycle_key],
            )
            self._complete_all()

    def _on_finish_requested(self):
        if self.context[self.name_key] == self.validation:
            return
        if self.validate_at_end and self.context[self.run_step_key] > 0:
            self._start_validation()
        else:
            if self.context[self.run_step_key] > 0 and self.context[self.name_key] == self.train:
                self.signal(
                    "phase_ended",
                    phase=self.train,
                    step=self.context[self.step_key],
                    cycle=self.context[self.cycle_key],
                )
            self._complete_all()

    def _start_validation(self):
        if self.context[self.due_key] or self.context[self.run_stop_key]:
            return
        if self.context[self.name_key] == self.train:
            self.signal(
                "phase_ended",
                phase=self.train,
                step=self.context[self.step_key],
                cycle=self.context[self.cycle_key],
            )
        self.context[self.name_key] = self.validation
        self.context[self.legacy_phase_key] = self.validation
        self.context[self.step_key] = 0
        self.context[self.due_key] = True
        self.context[self.legacy_validation_due_key] = True
        index = self.context[self.legacy_validation_index_key] + 1
        self.context[self.index_key] = index
        self.signal("phase_started", phase=self.validation, index=index, cycle=self.context[self.cycle_key])
        self.signal("validation_start", step=self.context[self.run_step_key], validation_index=index)

    def _finish_validation(self):
        index = self.context[self.index_key]
        self.signal(
            "phase_ended",
            phase=self.validation,
            step=self.context[self.step_key],
            index=index,
            cycle=self.context[self.cycle_key],
        )
        self.signal("validation_end", step=self.context[self.run_step_key], validation_index=index)
        self.context[self.due_key] = False
        self.context[self.legacy_validation_due_key] = False
        self.context[self.legacy_validation_index_key] = index
        if self.context[self.finish_requested_key]:
            self._complete_all()
            return

        cycle = self.context[self.cycle_key] + 1
        self.context[self.cycle_key] = cycle
        self.context[self.name_key] = self.train
        self.context[self.legacy_phase_key] = self.train
        self.context[self.step_key] = 0
        self.signal("phase_started", phase=self.train, index=index, cycle=cycle)
        self.signal("train_resume", step=self.context[self.run_step_key], cycle=cycle)

    def _complete_all(self):
        if self.completed:
            return
        self.completed = True
        self.context[self.due_key] = False
        self.context[self.legacy_validation_due_key] = False
        self.context[self.name_key] = "finished"
        self.context[self.legacy_phase_key] = "finished"
        self.signal("phases_completed", step=self.context[self.run_step_key])

    def state_dict(self):
        keys = (
            self.name_key,
            self.index_key,
            self.step_key,
            self.cycle_key,
            self.due_key,
            self.legacy_phase_key,
            self.legacy_validation_index_key,
            self.legacy_validation_due_key,
        )
        return {key: self.context[key] for key in keys}

    def load_state_dict(self, state):
        for key, value in state.items():
            self.context[key] = value
        self.completed = self.context[self.name_key] == "finished"
