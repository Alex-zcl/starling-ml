"""Публичная загрузка и редактирование конфигураций без создания ML-объектов."""
from copy import deepcopy
from pathlib import Path
import yaml


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_config(config):
    """Принимает dict, единый YAML или каталог четырёх YAML версии 0.2."""
    if isinstance(config, dict):
        return deepcopy(config)
    path = Path(config)
    if path.is_dir():
        return {key: load_yaml(path / f"{key}.yaml") for key in ("constants", "modules", "contracts", "pipeline")}
    value = load_yaml(path)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def save_config(config, path):
    """Сохраняет стандартный YAML; никаких скрытых runtime-объектов."""
    Path(path).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_engine(config):
    from .engine import Engine
    return Engine(config=config)
