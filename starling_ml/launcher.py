"""Независимые эксперименты в spawn-процессах; DDP — отдельный механизм."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path


def _run_one(item):
    index, config, output_root = item
    import torch
    from . import Engine, save_config
    from .checkpoint import save_checkpoint
    torch.set_num_threads(1)
    directory = Path(output_root).resolve() / f"experiment_{index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    # Отдельный cwd также изолирует относительные outputs модулей.
    import os
    os.chdir(directory)
    save_config(config, directory / "config.yaml")
    engine = Engine(config).run()
    save_checkpoint(engine, directory / "checkpoint.pt")
    return dict(index=index, directory=str(directory), step=engine.context.data.get("run.step"))


def run_experiments(configs, output_root, max_workers=2):
    """Каждый config задаёт свой seed/device; вызывайте под __main__ guard."""
    configs = [deepcopy(config) for config in configs]
    output_root = str(Path(output_root).resolve())
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_run_one, [(i, cfg, output_root) for i, cfg in enumerate(configs)]))
