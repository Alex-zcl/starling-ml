"""Строгая композиция независимых частей конфигурации эксперимента."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


class ConfigConflict(ValueError):
    """Два фрагмента неявно определяют одну и ту же часть конфигурации."""


@dataclass(frozen=True)
class ConfigFragment:
    """Именованный фрагмент, который можно безопасно объединить с другими.

    ``replace_modules`` делает замену намеренной и видимой. Обычное совпадение
    имён считается ошибкой: порядок аргументов compose не должен молча менять
    архитектуру эксперимента.
    """

    name: str
    config: Mapping[str, Any]
    replace_modules: frozenset[str] = field(default_factory=frozenset)
    replace_constants: frozenset[str] = field(default_factory=frozenset)
    pipeline_mode: str = "error"
    metadata_mode: str = "merge"

    def __post_init__(self):
        if self.pipeline_mode not in {"error", "append", "replace"}:
            raise ValueError("pipeline_mode must be error, append or replace")
        if self.metadata_mode not in {"merge", "replace"}:
            raise ValueError("metadata_mode must be merge or replace")


def fragment(
    name,
    *,
    constants=None,
    modules=None,
    contracts=None,
    pipeline=None,
    metadata=None,
    replace_modules=(),
    replace_constants=(),
    pipeline_mode="error",
    metadata_mode="merge",
):
    """Создаёт фрагмент без ML-объектов и скрытых побочных эффектов."""

    value = {
        "constants": dict(constants or {}),
        "modules": dict(modules or {}),
        "contracts": dict(contracts or {}),
        "pipeline": list(pipeline or []),
    }
    if metadata is not None:
        value["metadata"] = dict(metadata)
    return ConfigFragment(
        name=str(name),
        config=value,
        replace_modules=frozenset(replace_modules),
        replace_constants=frozenset(replace_constants),
        pipeline_mode=pipeline_mode,
        metadata_mode=metadata_mode,
    )


def recipe(name="segmentation", **options):
    """Возвращает готовый recipe как первый composable-фрагмент."""

    from .configs import get_config

    return ConfigFragment(name=f"recipe:{name}", config=get_config(name, **options))


def module_fragment(name, class_path, params=None, contract=None, *, replace=False):
    """Добавляет или явно заменяет один модуль вместе с его контрактом."""

    return fragment(
        f"module:{name}",
        modules={name: {"class": class_path, "params": dict(params or {})}},
        contracts={name: dict(contract or {})},
        replace_modules={name} if replace else (),
    )


def phase_module(
    name,
    class_path,
    *,
    phases=("train", "validation"),
    params=None,
    contract=None,
    overrides=None,
    separator="_",
):
    """Разворачивает один шаблон в независимый экземпляр на каждую фазу.

    Строка ``{phase}`` заменяется рекурсивно и в параметрах, и в контракте.
    Это удобно для stateful metrics/losses: описание остаётся одним, но
    накопители и владельцы context keys физически не разделяются между фазами.

    ``overrides`` позволяет фазе заменить ``class``, отдельные ``params`` или
    поля ``contract``. Поэтому validation loss может отличаться от train loss.
    """

    modules, contracts = {}, {}
    overrides = dict(overrides or {})
    for phase in phases:
        phase = str(phase)
        selected = dict(overrides.get(phase, {}))
        module_name = f"{phase}{separator}{name}"
        phase_params = deepcopy(dict(params or {}))
        phase_params.update(deepcopy(dict(selected.get("params", {}))))
        phase_contract = _merge_contracts(contract or {}, selected.get("contract", {}))
        modules[module_name] = {
            "class": selected.get("class", class_path),
            "params": _replace_phase(phase_params, phase),
        }
        contracts[module_name] = _replace_phase(phase_contract, phase)
    return fragment(f"phase-module:{name}", modules=modules, contracts=contracts)


def compose(*parts):
    """Собирает обычный Engine-config, запрещая неявные перезаписи."""

    result = {"constants": {}, "modules": {}, "contracts": {}, "pipeline": []}
    pipeline_owner = None

    for index, raw in enumerate(parts):
        part = raw if isinstance(raw, ConfigFragment) else ConfigFragment(
            name=f"config:{index}", config=raw
        )
        if not isinstance(part.config, Mapping):
            raise TypeError(f"{part.name}: config must be a mapping")
        cfg = deepcopy(dict(part.config))
        modules = cfg.get("modules", {})
        contracts = cfg.get("contracts", {})
        if set(modules) != set(contracts):
            raise ConfigConflict(
                f"{part.name}: modules and contracts differ: "
                f"modules={sorted(modules)}, contracts={sorted(contracts)}"
            )

        for key, value in cfg.get("constants", {}).items():
            if key in result["constants"] and result["constants"][key] != value:
                if key not in part.replace_constants:
                    raise ConfigConflict(
                        f"{part.name}: constant '{key}' is already defined; "
                        "declare replace_constants explicitly"
                    )
            result["constants"][key] = value

        for name, value in modules.items():
            if name in result["modules"] and name not in part.replace_modules:
                raise ConfigConflict(
                    f"{part.name}: module '{name}' is already defined; "
                    "use module_fragment(..., replace=True) or replace_modules"
                )
            result["modules"][name] = value
            result["contracts"][name] = contracts[name]

        incoming_pipeline = cfg.get("pipeline", [])
        if incoming_pipeline:
            if not result["pipeline"] or part.pipeline_mode == "replace":
                result["pipeline"] = incoming_pipeline
                pipeline_owner = part.name
            elif part.pipeline_mode == "append":
                result["pipeline"].extend(incoming_pipeline)
            else:
                raise ConfigConflict(
                    f"{part.name}: pipeline is already provided by {pipeline_owner}; "
                    "select pipeline_mode='append' or 'replace' explicitly"
                )

        if "metadata" in cfg:
            if part.metadata_mode == "replace" or "metadata" not in result:
                result["metadata"] = cfg["metadata"]
            else:
                _merge_mapping(result["metadata"], cfg["metadata"], part.name, "metadata")

        known = {"constants", "modules", "contracts", "pipeline", "metadata"}
        for key, value in cfg.items():
            if key in known:
                continue
            if key in result and result[key] != value:
                raise ConfigConflict(f"{part.name}: top-level key '{key}' conflicts")
            result[key] = value

    return result


def _merge_mapping(target, incoming, owner, path):
    if not isinstance(target, dict) or not isinstance(incoming, Mapping):
        if target != incoming:
            raise ConfigConflict(f"{owner}: '{path}' conflicts")
        return
    for key, value in incoming.items():
        location = f"{path}.{key}"
        if key not in target:
            target[key] = deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _merge_mapping(target[key], value, owner, location)
        elif target[key] != value:
            raise ConfigConflict(f"{owner}: '{location}' conflicts")


def _replace_phase(value, phase):
    if isinstance(value, str):
        return value.replace("{phase}", phase)
    if isinstance(value, Mapping):
        return {key: _replace_phase(item, phase) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_phase(item, phase) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_phase(item, phase) for item in value)
    return value


def _merge_contracts(base, override):
    result = {key: list(values) for key, values in dict(base).items()}
    for key, values in dict(override or {}).items():
        combined = [*result.get(key, []), *list(values)]
        result[key] = list(dict.fromkeys(combined))
    return result
