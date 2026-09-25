"""Composable monitoring profiles для любых типов экспериментов."""

from .composition import fragment


def monitoring_profile(
    *profiles,
    run_name="experiment",
    root="runs",
    scalars=None,
    phase_scalars=None,
    validation_total=None,
    progress_mode="global",
    train_total=None,
    clearml_project="Starling ML",
    clearml_task=None,
):
    """Создаёт независимый monitoring-фрагмент.

    Профили: ``none``, ``console``, ``progress``, ``text``, ``tensorboard``,
    ``clearml``, ``standard`` (console+progress+text) и ``tracking``
    (standard+tensorboard). ClearML всегда подключается явно.
    """

    selected = _expand_profiles(profiles or ("console",))
    if not selected:
        return fragment("monitoring:none")

    scalars = dict(scalars or {"loss": "train.loss"})
    phase_scalars = {
        phase: dict(values) for phase, values in (phase_scalars or {}).items()
    }
    scalar_keys = sorted(
        {"run.step", *scalars.values(), *(key for values in phase_scalars.values() for key in values.values())}
    )
    modules, contracts = {}, {}

    needs_directory = bool(selected & {"text", "tensorboard"})
    if needs_directory:
        modules["run_directory"] = {
            "class": "starling_ml.modules.monitoring.files.RunDirectoryManager",
            "params": {"root": root, "name": run_name},
        }
        contracts["run_directory"] = {
            "creates": [
                "run.directory",
                "run.log_file",
                "run.tensorboard_directory",
                "run.checkpoint_directory",
                "run.artifact_directory",
            ]
        }

    if "console" in selected:
        modules["console_logger"] = {
            "class": "starling_ml.modules.monitoring.basic.ConsoleLogger",
            "params": {
                "scalars": scalars,
                "phase_scalars": phase_scalars,
                "signals": ["train_step_end", "metrics_ready", "run_end"],
            },
        }
        contracts["console_logger"] = {"reads": scalar_keys}

    if "progress" in selected:
        params = {
            "total": "$const:max_steps",
            "train_scalars": scalars,
            "validation_scalars": phase_scalars.get("validation", {}),
            "mode": progress_mode,
        }
        if validation_total is not None:
            params["validation_total"] = validation_total
        if train_total is not None:
            params["train_total"] = train_total
        modules["progress"] = {
            "class": "starling_ml.modules.monitoring.progress.TqdmReporter",
            "params": params,
        }
        contracts["progress"] = {"reads": scalar_keys}

    if "text" in selected:
        modules["text_logger"] = {
            "class": "starling_ml.modules.monitoring.files.TextFileLogger",
            "params": {
                "path": "$ctx:run.log_file",
                "scalars": scalars,
                "phase_scalars": phase_scalars,
            },
        }
        contracts["text_logger"] = {"reads": ["run.log_file", *scalar_keys]}

    if "tensorboard" in selected:
        modules["tensorboard_logger"] = {
            "class": "starling_ml.integrations.tensorboard.TensorBoardLogger",
            "params": {
                "log_dir": "$ctx:run.tensorboard_directory",
                "scalars": scalars,
                "phase_scalars": phase_scalars,
            },
        }
        contracts["tensorboard_logger"] = {
            "reads": ["run.tensorboard_directory", *scalar_keys]
        }

    if "clearml" in selected:
        modules["clearml_logger"] = {
            "class": "starling_ml.integrations.clearml.ClearMLLogger",
            "params": {
                "project_name": clearml_project,
                "task_name": clearml_task or run_name,
                "scalars": scalars,
                "phase_scalars": phase_scalars,
            },
        }
        contracts["clearml_logger"] = {"reads": scalar_keys}

    return fragment(
        "monitoring:" + "+".join(sorted(selected)),
        modules=modules,
        contracts=contracts,
    )


def _expand_profiles(profiles):
    aliases = {
        "none": set(),
        "console": {"console"},
        "progress": {"progress"},
        "text": {"text"},
        "tensorboard": {"tensorboard"},
        "clearml": {"clearml"},
        "standard": {"console", "progress", "text"},
        "tracking": {"console", "progress", "text", "tensorboard"},
    }
    result = set()
    for profile in profiles:
        if profile not in aliases:
            raise ValueError(f"Unknown monitoring profile: {profile}")
        if profile == "none":
            if len(profiles) > 1:
                raise ValueError("monitoring profile 'none' cannot be combined")
            return set()
        result.update(aliases[profile])
    return result
