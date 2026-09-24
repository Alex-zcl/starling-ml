"""Управляемый общий контекст эксперимента."""

from types import MappingProxyType


def freeze(value):
    """Замораживает только конфигурационные контейнеры, не копируя ML-объекты."""
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


class Context:
    """Хранит константы и runtime-данные с явными правами модулей.

    Контекст остаётся простым словарём по смыслу, но не позволяет модулю
    незаметно читать или менять чужое состояние. Это сохраняет гибкость
    общего контекста и делает зависимости проверяемыми.
    """

    def __init__(self, constants, contracts):
        self.constants = freeze(dict(constants))
        self._data = {}
        self._contracts = contracts
        self._owners = {}

        # У каждого runtime-ключа должен быть ровно один создатель.
        # Так источник значения всегда можно определить по конфигурации.
        for module, contract in contracts.items():
            for key in contract.get("creates", []):
                if key in self._owners:
                    owner = self._owners[key]
                    raise ValueError(f"'{key}' is created by both {owner} and {module}")
                self._owners[key] = module

    @property
    def data(self):
        """Возвращает read-only представление всех runtime-данных."""
        return MappingProxyType(self._data)

    def view(self, module):
        """Создаёт ограниченное представление контекста для одного модуля."""
        return ContextView(self, module)


class ContextView:
    """Проверяет права конкретного модуля при каждом чтении и записи."""

    def __init__(self, context, module):
        self._context = context
        self.module = module

    def const(self, key):
        """Читает неизменяемую константу эксперимента."""
        return self._context.constants[key]

    def has(self, key):
        """Проверяет наличие разрешённого ключа без чтения его значения."""
        self._check_read_permission(key)
        return key in self._context._data

    def __getitem__(self, key):
        self._check_read_permission(key)
        if key not in self._context._data:
            raise KeyError(f"'{key}' does not exist yet")
        return self._context._data[key]

    def mutable(self, key):
        """Проверяет объявленное право изменить объект по ссылке.

        Обычный reads не является защитной proxy: пользовательский Python-код
        обязан соблюдать контракт. Встроенные адаптеры объявляют mutates явно.
        """
        contract = self._context._contracts[self.module]
        if key not in {*contract.get("creates", []), *contract.get("updates", []), *contract.get("mutates", [])}:
            raise PermissionError(f"{self.module} cannot mutate '{key}'")
        return self[key]

    def get(self, key, default=None):
        """Возвращает значение или default, если разрешённый ключ ещё не создан."""
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        owner = self._context._owners.get(key)
        updates = set(self._context._contracts[self.module].get("updates", []))

        # Создать значение может только его владелец. После создания владелец
        # может обновлять его без дублирования ключа в секции updates.
        if key not in self._context._data:
            if owner != self.module:
                raise PermissionError(f"{self.module} cannot create '{key}'")
        elif owner != self.module and key not in updates:
            raise PermissionError(f"{self.module} cannot update '{key}'")

        self._context._data[key] = value

    def _check_read_permission(self, key):
        """Проверяет чтение отдельно, чтобы has/get не обходили контракт."""
        contract = self._context._contracts[self.module]
        allowed = set(contract.get("reads", []))
        allowed.update(contract.get("creates", []))
        allowed.update(contract.get("updates", []))
        allowed.update(contract.get("mutates", []))
        if key not in allowed:
            raise PermissionError(f"{self.module} cannot read '{key}'")
