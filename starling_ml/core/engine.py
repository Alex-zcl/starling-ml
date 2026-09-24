"""Небольшой интерпретатор конфигурации и runtime-pipeline."""

import importlib

from .context import Context


class Engine:
    """Создаёт модули, выводит setup-порядок и исполняет простой DSL.

    Ядро не знает, что такое loss, optimizer, эпоха или валидация. Оно умеет
    только вызвать модуль, отправить сигнал и исполнить общие конструкции
    управления потоком. Поэтому новые ML-сценарии добавляются модулями.
    """

    def __init__(self, constants=None, modules=None, contracts=None, pipeline=None, *, config=None, check=True):
        # Единый config удобен пользователю; старый явный API сохраняется.
        if config is not None or (modules is None and constants is not None):
            from .config import read_config
            config = read_config(config if config is not None else constants)
            self.config = config
            constants, modules, contracts, pipeline = (config.get(k, {} if k != "pipeline" else []) for k in ("constants", "modules", "contracts", "pipeline"))
        else:
            self.config = dict(constants=constants or {}, modules=modules or {}, contracts=contracts or {}, pipeline=pipeline or [])
        self._is_setup = False
        self._closed = False
        self.check = check
        self.module_config = modules or {}
        self.contracts = contracts or {}
        self.pipeline = pipeline or []
        self.context = Context(constants or {}, contracts or {})
        self.modules = {}

        if self.check:
            from ..analysis import analyze_config
            self.analysis = analyze_config(self.config)
            self.analysis.raise_for_errors()
        self._validate()
        self._build_modules()

    def setup(self):
        """Инициализирует модули в автоматически найденном порядке."""
        if self._is_setup:
            return self
        for name in self._setup_order():
            params = self._resolve(self.module_config[name].get("params", {}))
            self.modules[name].setup(**params)
        self._is_setup = True
        for module in self.modules.values():
            module.ready()
        return self

    def run(self):
        """Исполняет pipeline без скрытой перестановки runtime-шагов."""
        if self._closed:
            raise RuntimeError("Engine already closed; create a new Engine to resume")
        try:
            self.setup()
            self._run_steps(self.pipeline)
        except BaseException as exc:
            # Ошибка не маскируется ошибкой cleanup; задача close — освободить ресурсы.
            for module in reversed(list(self.modules.values())):
                try:
                    module.close()
                except Exception as cleanup_error:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"Cleanup {module.name}: {cleanup_error}")
            self._closed = True
            raise
        else:
            self.close()
        return self

    def state_dict(self):
        """Состояние принадлежит модулям: ядро не знает optimizer или scheduler."""
        return {name: module.state_dict() for name, module in self.modules.items()}

    def load_state_dict(self, state, strict=True):
        """Восстанавливает модули после setup, сохраняя явные границы владения."""
        if strict and set(state) != set(self.modules):
            raise ValueError("Checkpoint module names differ from configuration")
        for name, value in state.items():
            if name in self.modules:
                self.modules[name].load_state_dict(value)

    def close(self):
        """Освобождает ресурсы всех модулей даже при ошибке одного из них."""
        if self._closed:
            return
        self._closed = True
        errors = []
        for module in reversed(list(self.modules.values())):
            try:
                module.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def signal(self, signal, source=None, **payload):
        """Синхронно рассылает событие всем модулям."""
        for module in list(self.modules.values()):
            module.reaction(signal, source=source, **payload)

    # Этот блок проверяет только структурные ошибки, которые можно найти до
    # запуска. Фактическое наличие runtime-значения по-прежнему проверяет Context.
    def _validate(self):
        module_names = set(self.module_config)
        contract_names = set(self.contracts)
        if module_names != contract_names:
            raise ValueError(
                "Modules and contracts differ: "
                f"modules={sorted(module_names)}, contracts={sorted(contract_names)}"
            )

        owners = self.context._owners
        for module, contract in self.contracts.items():
            for key in contract.get("reads", []):
                if key not in owners:
                    raise ValueError(f"{module} reads '{key}', but nobody creates it")
            for key in [*contract.get("updates", []), *contract.get("mutates", [])]:
                if key not in owners:
                    raise ValueError(f"{module} updates '{key}', but nobody creates it")

        for module, cfg in self.module_config.items():
            contract = self.contracts[module]
            allowed = set(contract.get("reads", []))
            allowed.update(contract.get("creates", []))
            allowed.update(contract.get("updates", []))
            allowed.update(contract.get("mutates", []))
            for key in self._context_refs(cfg.get("params", {})):
                if key not in owners:
                    raise ValueError(f"{module} requires $ctx:{key}, but nobody creates it")
                if key not in allowed:
                    raise ValueError(
                        f"{module} receives $ctx:{key} in setup, but its contract cannot read it"
                    )

        self._validate_steps(self.pipeline)

    def _validate_steps(self, steps):
        """Рекурсивно проверяет имена модулей в pipeline."""
        for step in steps:
            if not isinstance(step, dict) or len(step) != 1 or not set(step) <= {"call", "signal", "loop", "while", "when", "iterate"}:
                raise ValueError(f"Invalid pipeline step: {step!r}")
            if "call" in step and step["call"] not in self.module_config:
                raise ValueError(f"Unknown module in pipeline: {step['call']}")
            if "iterate" in step:
                source = step["iterate"]["source"]
                if source not in self.module_config:
                    raise ValueError(f"Unknown iterate source: {source}")
                self._validate_steps(step["iterate"].get("body", []))
            for name in ("loop", "while", "when"):
                if name in step:
                    self._validate_steps(step[name].get("body", []))

    def _build_modules(self):
        """Создаёт лёгкие Python-обёртки до тяжёлого setup."""
        for name, cfg in self.module_config.items():
            cls = self._import(cfg["class"])
            module = cls(self.context.view(name))
            module.name = name
            module._emit = self.signal
            module._snapshot = self.state_dict
            module._restore = self.load_state_dict
            self.modules[name] = module

    def _setup_order(self):
        """Строит topological order по ссылкам $ctx в setup-параметрах."""
        owners = self.context._owners
        deps = {
            name: {
                owners[key]
                for key in self._context_refs(cfg.get("params", {}))
                if owners[key] != name
            }
            for name, cfg in self.module_config.items()
        }

        order = []
        pending = set(self.modules)
        while pending:
            ready = [name for name in self.modules if name in pending and deps[name] <= set(order)]
            if not ready:
                details = {name: sorted(deps[name] & pending) for name in sorted(pending)}
                raise ValueError(f"Setup dependency cycle: {details}")
            order.extend(ready)
            pending.difference_update(ready)
        return order

    # Pipeline DSL специально мал: call/signal/loop сохранены, а while/when/iterate
    # являются общими конструкциями и не привязывают Engine к конкретной ML-задаче.
    def _run_steps(self, steps):
        for step in steps:
            if "call" in step:
                self.modules[step["call"]]()

            elif "signal" in step:
                signal = step["signal"]
                if isinstance(signal, str):
                    self.signal(signal, source="Engine")
                else:
                    payload = self._resolve(signal.get("payload", {}))
                    self.signal(signal["name"], source="Engine", **payload)

            elif "loop" in step:
                loop = step["loop"]
                for _ in range(int(self._resolve(loop["count"]))):
                    self._run_steps(loop["body"])

            elif "while" in step:
                block = step["while"]
                iterations = 0
                limit = block.get("max_iterations")
                limit = int(self._resolve(limit)) if limit is not None else None
                while self._condition(block):
                    if limit is not None and iterations >= limit:
                        raise RuntimeError("Pipeline while reached max_iterations")
                    self._run_steps(block["body"])
                    iterations += 1

            elif "when" in step:
                block = step["when"]
                if self._condition(block):
                    self._run_steps(block["body"])

            elif "iterate" in step:
                block = step["iterate"]
                source = self.modules[block["source"]]
                while True:
                    try:
                        source()
                    except StopIteration:
                        break
                    self._run_steps(block.get("body", []))

            else:
                raise ValueError(f"Unknown pipeline step: {step}")

    def _condition(self, block):
        """Вычисляет простое условие без собственного языка выражений."""
        value = self._resolve(block["condition"])
        if "equals" in block:
            return value == self._resolve(block["equals"])
        if "not_equals" in block:
            return value != self._resolve(block["not_equals"])
        return bool(value)

    def _resolve(self, value):
        """Разрешает ссылки на константы и уже созданное состояние."""
        if isinstance(value, str):
            if value.startswith("$const:"):
                return self.context.constants[value[7:]]
            if value.startswith("$ctx:"):
                key = value[5:]
                if key not in self.context._data:
                    raise KeyError(f"Cannot resolve {value}: value does not exist yet")
                return self.context._data[key]
            return value
        if isinstance(value, (list, tuple)):
            return [self._resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        return value

    def _context_refs(self, value):
        """Собирает $ctx-ссылки для построения setup-графа."""
        if isinstance(value, str) and value.startswith("$ctx:"):
            return [value[5:]]
        if isinstance(value, (list, tuple)):
            return [key for item in value for key in self._context_refs(item)]
        if isinstance(value, dict):
            return [key for item in value.values() for key in self._context_refs(item)]
        return []

    @staticmethod
    def _import(path):
        """Импортирует класс по строке package.module.Object."""
        module_name, object_name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), object_name)
