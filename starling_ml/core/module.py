"""Минимальный базовый класс runtime-модуля."""


class Module:
    """Единица работы движка с необязательными setup/call/reaction.

    Класс намеренно почти пустой: новый модуль должен описывать только свою
    работу, а не наследовать сложный жизненный цикл большого Trainer-класса.
    """

    def __init__(self, context):
        self.context = context
        self.name = None
        self._emit = None

    def setup(self, **params):
        """Создаёт тяжёлые объекты после разрешения setup-зависимостей."""

    def __call__(self):
        """Выполняет одну runtime-операцию модуля."""

    def reaction(self, signal, source=None, **payload):
        """Необязательно реагирует на широковещательное событие."""

    def signal(self, signal, **payload):
        """Рассылает событие, не зная список его получателей."""
        self._emit(signal, source=self.name, **payload)

    def ready(self):
        """Вызывается после setup всех модулей; подходит для checkpoint restore."""

    def close(self):
        """Освобождает ресурсы; реализация должна быть идемпотентной."""

    def state_dict(self):
        """Stateful-модуль явно описывает своё состояние вместо pickle всего объекта."""
        if not hasattr(self.context, "_context"):
            return {}
        creates = self.context._context._contracts[self.name].get("creates", [])
        return {key: self.context[key].state_dict() for key in creates
                if self.context.has(key) and hasattr(self.context[key], "state_dict")}


    def load_state_dict(self, state):
        for key, value in state.items():
            obj = self.context[key]
            if not hasattr(obj, "load_state_dict"):
                raise ValueError(f"{type(self).__name__} cannot restore {key}")
            obj.load_state_dict(value)
