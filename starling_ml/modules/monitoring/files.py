"""Стандартные run-directory и текстовый лог без внешних зависимостей."""

from datetime import datetime, timezone
from pathlib import Path

from ...core.module import Module
from ...ops.tensors import scalar


class RunDirectoryManager(Module):
    """Создаёт единое дерево путей для логов, checkpoints и artifacts."""

    def setup(
        self,
        root="runs",
        name=None,
        create=True,
        output="run.directory",
        log_file="run.log_file",
        tensorboard="run.tensorboard_directory",
        checkpoints="run.checkpoint_directory",
        artifacts="run.artifact_directory",
    ):
        if name is None:
            name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        directory = Path(root) / str(name)
        paths = {
            output: directory,
            log_file: directory / "experiment.log",
            tensorboard: directory / "tensorboard",
            checkpoints: directory / "checkpoints",
            artifacts: directory / "artifacts",
        }
        if create:
            directory.mkdir(parents=True, exist_ok=True)
            paths[tensorboard].mkdir(exist_ok=True)
            paths[checkpoints].mkdir(exist_ok=True)
            paths[artifacts].mkdir(exist_ok=True)
        for key, value in paths.items():
            self.context[key] = str(value)


class TextFileLogger(Module):
    """Пишет phase-aware scalar-события в обычный UTF-8 файл."""

    def __init__(self, context):
        super().__init__(context)
        self.stream = None

    def setup(
        self,
        path,
        scalars=None,
        phase_scalars=None,
        signals=("train_step_end", "metrics_ready", "run_end"),
        step_key="run.step",
        flush=True,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")
        self.scalars = dict(scalars or {})
        self.phase_scalars = {
            phase: dict(values) for phase, values in (phase_scalars or {}).items()
        }
        self.signals = set(signals)
        self.step_key = step_key
        self.flush = bool(flush)

    def reaction(self, signal, source=None, **payload):
        if self.stream is None or signal not in self.signals:
            return
        phase = payload.get("phase")
        mapping = self.phase_scalars.get(phase, self.scalars)
        values = []
        for name, key in mapping.items():
            value = self.context.get(key)
            if value is not None:
                values.append(f"{name}={scalar(value):.6f}")
        timestamp = datetime.now(timezone.utc).isoformat()
        header = f"{timestamp} signal={signal} step={int(self.context.get(self.step_key, 0))}"
        if phase:
            header += f" phase={phase}"
        self.stream.write(header + (" " + " ".join(values) if values else "") + "\n")
        if self.flush:
            self.stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
