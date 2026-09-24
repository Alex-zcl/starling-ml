"""Объяснимый preflight: диагностика не меняет конфигурацию или pipeline."""
from dataclasses import dataclass, field
import importlib
import inspect
from collections.abc import Mapping
from .core.config import read_config


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    location: str
    message: str


@dataclass
class AnalysisReport:
    issues: list = field(default_factory=list)
    objectives: list = field(default_factory=list)

    @property
    def errors(self):
        return [item for item in self.issues if item.severity == "error"]

    @property
    def warnings(self):
        return [item for item in self.issues if item.severity == "warning"]

    @property
    def ok(self):
        return not self.errors

    def add(self, severity, code, location, message):
        self.issues.append(Issue(severity, code, location, message))

    def raise_for_errors(self):
        if self.errors:
            raise ValueError(str(self))
        return self

    def __str__(self):
        labels = {"error": "ОШИБКА", "warning": "ПРЕДУПРЕЖДЕНИЕ", "pending": "ПРОВЕРИТЬ"}
        lines = [f"Анализ конфигурации: ошибок {len(self.errors)}, предупреждений {len(self.warnings)}."]
        lines += [f"[{labels[x.severity]} {x.code}] {x.location}: {x.message}" for x in self.issues]
        lines += ["Objective: " + text for text in self.objectives]
        return "\n".join(lines)


def _refs(value, prefix):
    if isinstance(value, str):
        return [value[len(prefix):]] if value.startswith(prefix) else []
    if isinstance(value, Mapping):
        return [key for v in value.values() for key in _refs(v, prefix)]
    if isinstance(value, (list, tuple)):
        return [key for v in value for key in _refs(v, prefix)]
    return []


def _resolve_constants(value, constants):
    if isinstance(value, str) and value.startswith("$const:"):
        return constants.get(value[7:], value)
    if isinstance(value, Mapping):
        return {k: _resolve_constants(v, constants) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_constants(v, constants) for v in value]
    return value


def analyze_config(config, *, metadata=None, batch=None):
    """Три стадии: структура; metadata dataset; опубликованные runtime tensors.

    Импортирует классы для проверки signature, но не вызывает setup/forward и не
    потребляет dataloader. Пользовательские import paths должны быть доверенными.
    """
    report = AnalysisReport()
    add = report.add
    try:
        config = read_config(config)
    except Exception as exc:
        add("error", "CONFIG", "config", str(exc))
        return report
    allowed = {"constants", "modules", "contracts", "pipeline", "metadata"}
    for key in set(config) - allowed:
        add("error", "UNKNOWN_KEY", key, "Неизвестный раздел конфигурации.")
    for key in ("constants", "modules", "contracts", "metadata"):
        if key in config and not isinstance(config[key], Mapping):
            add("error", "MAPPING", key, "Ожидается словарь.")
    if report.errors:
        return report
    modules, contracts = config.get("modules", {}), config.get("contracts", {})
    constants = config.get("constants", {})
    if set(modules) != set(contracts):
        add("error", "CONTRACTS", "contracts", "Для каждого модуля нужен ровно один контракт.")
    owners = {}
    for name, contract in contracts.items():
        if not isinstance(contract, Mapping):
            add("error", "CONTRACT", name, "Контракт должен быть словарём.")
            continue
        for kind, keys in contract.items():
            if kind not in {"creates", "reads", "updates", "mutates"}:
                add("error", "CONTRACT_KEY", name, f"Неизвестное право {kind}.")
            if not isinstance(keys, (list, tuple)) or not all(isinstance(k, str) for k in keys):
                add("error", "CONTRACT_KEYS", name, "Ключи контракта — список строк.")
                continue
            if kind == "creates":
                for key in keys:
                    if key in owners:
                        add("error", "DUPLICATE_OWNER", name, f"{key} уже создаёт {owners[key]}.")
                    owners[key] = name
    if report.errors:
        return report
    dependencies = {}
    resolved = {}
    for name, spec in modules.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("class"), str) or not isinstance(spec.get("params", {}), Mapping):
            add("error", "MODULE", name, "Нужны class: import path и params: словарь.")
            continue
        for key in set(spec) - {"class", "params"}:
            add("error", "MODULE_KEY", name, f"Неизвестный параметр модуля {key}.")
        params = _resolve_constants(spec.get("params", {}), constants)
        resolved[name] = params
        contract = contracts.get(name, {})
        readable = set().union(*(set(contract.get(k, [])) for k in ("reads", "creates", "updates", "mutates")))
        for key in readable - set(owners):
            add("error", "NO_OWNER", name, f"Никто не создаёт {key}.")
        refs = _refs(params, "$ctx:")
        for key in refs:
            if key not in readable:
                add("error", "SETUP_PERMISSION", name, f"Нет права читать $ctx:{key}.")
            if key not in owners:
                add("error", "SETUP_REFERENCE", name, f"Никто не создаёт $ctx:{key}.")
        dependencies[name] = {owners[k] for k in refs if k in owners and owners[k] != name}
        for key in _refs(spec.get("params", {}), "$const:"):
            if key not in constants:
                add("error", "CONSTANT", name, f"Неизвестная константа {key}.")
        try:
            path, cls_name = spec["class"].rsplit(".", 1)
            cls = getattr(importlib.import_module(path), cls_name)
            from .core.module import Module
            if not inspect.isclass(cls) or not issubclass(cls, Module):
                raise TypeError("class должен наследовать Module")
            inspect.signature(cls.setup).bind(None, **params)
        except Exception as exc:
            add("error", "SETUP_SIGNATURE", name, str(exc))
        try:
            _settings(report, name, spec["class"].rsplit(".", 1)[-1], params)
        except (TypeError, ValueError) as exc:
            add("error", "PARAMETER_TYPE", name, str(exc))
        _runtime_references(report, name, spec["class"].rsplit(".", 1)[-1], params, contract, owners)
        # Объект по ссылке не защищён proxy; встроенные мутации должны быть видны.
        if spec["class"].rsplit(".", 1)[-1] in {"Forward", "OptimizationManager", "ModelMode", "ModelToDevice", "DistributedModel"}:
            model = params.get("model", "")
            if isinstance(model, str) and model.startswith("$ctx:") and model[5:] not in contract.get("mutates", []):
                add("warning", "MUTATES", name, "Модуль меняет модель/её режим; объявите model key в mutates.")
    weight_dependencies = {}
    for name, spec in modules.items():
        if isinstance(spec, Mapping) and spec.get("class", "").endswith("WeightProduct"):
            weight_dependencies[name] = {owners[k] for k in spec.get("params", {}).get("inputs", []) if k in owners}
    weight_pending = set(weight_dependencies)
    while weight_pending:
        ready = {name for name in weight_pending if not (weight_dependencies[name] & weight_pending)}
        if not ready:
            add("error", "WEIGHT_CYCLE", "modules", f"Цикл WeightProduct: {sorted(weight_pending)}.")
            break
        weight_pending -= ready
    pending = set(dependencies)
    done = set()
    while pending:
        ready = {n for n in pending if dependencies[n] <= done}
        if not ready:
            add("error", "SETUP_CYCLE", "modules", f"Цикл setup: {sorted(pending)}.")
            break
        done.update(ready)
        pending -= ready

    def steps(items, location="pipeline"):
        if not isinstance(items, (list, tuple)):
            add("error", "PIPELINE", location, "Ожидается список шагов.")
            return
        for index, step in enumerate(items):
            loc = f"{location}[{index}]"
            if not isinstance(step, Mapping) or len(step) != 1:
                add("error", "STEP", loc, "Шаг содержит ровно одну операцию.")
                continue
            op, value = next(iter(step.items()))
            if op == "call":
                if not isinstance(value, str) or value not in modules:
                    add("error", "CALL", loc, f"Неизвестный модуль {value}.")
            elif op == "signal":
                if not isinstance(value, str) and not (isinstance(value, Mapping) and isinstance(value.get("name"), str)):
                    add("error", "SIGNAL", loc, "Нужно имя события или {name, payload}.")
            elif op in {"loop", "while", "when", "iterate"}:
                required = "count" if op == "loop" else "source" if op == "iterate" else "condition"
                if not isinstance(value, Mapping) or required not in value or "body" not in value:
                    add("error", "BLOCK", loc, f"Нужны {required} и body.")
                    continue
                if op == "iterate" and value["source"] not in modules:
                    add("error", "SOURCE", loc, "Неизвестный iterate source.")
                if op == "while" and "max_iterations" not in value:
                    add("warning", "UNBOUNDED_LOOP", loc, "Нет max_iterations; завершение зависит от runtime condition.")
                steps(value["body"], loc + ".body")
            else:
                add("error", "OPERATION", loc, f"Неизвестная операция {op}.")
    steps(config.get("pipeline", []))
    for key in _refs(config.get("pipeline", []), "$ctx:"):
        if key not in owners:
            add("error", "PIPELINE_REFERENCE", "pipeline", f"Никто не создаёт {key}.")
    for key in _refs(config.get("pipeline", []), "$const:"):
        if key not in constants:
            add("error", "CONSTANT", "pipeline", f"Неизвестная константа {key}.")
    # Происхождение коэффициента прослеживается по владельцу context key.
    for name, params in resolved.items():
        sources = []
        for role in ("label_cost", "element_weight", "positive_cost", "negative_cost", "class_weight", "case_class_weight", "sample_weight", "present_weight", "empty_weight", "empty_cost"):
            value = params.get(role)
            if value is None:
                continue
            if isinstance(value, str) and value in owners:
                producer = owners[value]
                origin = modules[producer]["class"].rsplit(".", 1)[-1]
                source_params = resolved.get(producer, {})
                detail = ""
                if origin == "DatasetWeights":
                    detail = f"; level={source_params.get('level', 'element')}; balance={source_params.get('balance', 'class')}; unit={('element' if source_params.get('level', 'element') == 'element' else source_params.get('unit', 'study'))}; gamma={source_params.get('gamma', .5)}"
                sources.append(f"{role} ← {producer} ({origin}{detail})")
            else:
                sources.append(f"{role} = {value!r}")
        if sources and ".losses." in modules[name]["class"]:
            report.objectives.append(name + ": " + "; ".join(sources))
            frequency_sources = [owners[v] for v in params.values() if isinstance(v, str) and v in owners and modules[owners[v]]["class"].endswith("DatasetWeights")]
            if len(set(frequency_sources)) > 1:
                add("warning", "MULTIPLE_COMPENSATION", name, "Несколько частотных источников: проверьте, что один дисбаланс не компенсируется повторно.")
    _metadata(report, metadata if metadata is not None else config.get("metadata"), resolved)
    if batch is None:
        add("pending", "RUNTIME", "batch", "Shapes, dtype, masks и scalar losses проверяются на первом вызове; можно передать batch=engine.context.data.")
    else:
        _batch(report, batch, modules, resolved)
    return report


def _settings(report, name, kind, p):
    add = report.add
    overlap = kind in {"DiceLoss", "IoULoss", "TverskyLoss", "OverlapLoss", "GeneralizedDiceLoss", "ForegroundDiceLoss"}
    binary = kind in {"BCEWithLogits", "BinaryFocalWithLogits", "FocalLoss"}
    categorical = kind in {"CrossEntropy", "MulticlassFocalLoss"}
    if overlap or binary or categorical or kind in {"MSELoss", "L1Loss"}:
        reduction = p.get("reduction", "element_mean" if categorical else "pair_mean")
        options = {"element_mean", "sample_mean"} if categorical else {"pair_mean", "sample_mean", "class_macro"} if overlap else {"pair_mean", "sample_mean", "class_macro", "element_mean"}
        if reduction not in options:
            add("error", "REDUCTION", name, f"Недопустимая reduction {reduction}.")
        norm = p.get("element_normalization", "weight_sum")
        if norm not in {"weight_sum", "valid_count"}:
            add("error", "NORMALIZATION", name, "element_normalization: weight_sum или valid_count.")
        denominator = "сумма importance" if norm == "weight_sum" else "число валидных элементов"
        report.objectives.append(f"{name}: {kind}; reducer={reduction}; " + ("TP/FP/FN внутри ratio, внешние веса нормируются." if overlap else f"element denominator={denominator}; costs только в числителе."))
        if p.get("element_weight") is not None and norm == "weight_sum":
            add("warning", "WEIGHT_CANCELLATION", name, "Постоянный element importance сокращается при нормировке; это не абсолютная цена ошибки.")
        if overlap and p.get("element_weight") is not None:
            add("warning", "OVERLAP_CANCELLATION", name, "Постоянный множитель TP/FP/FN сокращается при smooth=0; используйте target-class map для различий между пикселями.")
        if overlap and p.get("empty_target", "ignore") == "ignore" and (p.get("case_class_weight") is not None or p.get("empty_weight") is not None or p.get("empty_cost", 1) != 1):
            add("warning", "EMPTY_IGNORED", name, "Пустые masks исключены: empty case weight/cost не действует.")
        if overlap and (p.get("present_weight") is not None or p.get("empty_weight") is not None):
            add("warning", "PRESENCE_NORMALIZATION", name, "present/empty нормируются в reducer; в однородной группе общий factor сокращается.")
        if overlap and p.get("aggregation", "sample") not in {"sample", "batch", "batch_stats"}:
            add("error", "AGGREGATION", name, "Неизвестный способ объединения overlap stats.")
        if overlap and p.get("mode", "multilabel") not in {"binary", "multilabel", "multiclass"}:
            add("error", "TASK_MODE", name, "Неизвестный task mode.")
        if overlap and p.get("empty_target", "ignore") not in {"ignore", "standard", "penalize", "false_positive"}:
            add("error", "EMPTY_POLICY", name, "Неизвестная empty policy.")
        if overlap and p.get("aggregation", "sample") in {"batch", "batch_stats"}:
            add("warning", "BATCH_STATS", name, "Сначала суммируются TP/FP/FN; это не среднее sample Dice. Sample/case weights входят внутрь stats.")
        for key in ("positive_cost", "negative_cost", "label_cost", "empty_cost", "class_weight", "sample_weight"):
            value = p.get(key)
            if isinstance(value, (int, float)) and (value < 0 or not __import__("math").isfinite(value)):
                add("error", "WEIGHT_VALUE", name, f"{key} должен быть конечным и неотрицательным.")
    if kind == "OptimizationManager":
        if p.get("accumulation", 1) < 1:
            add("error", "ACCUMULATION", name, "accumulation должен быть >= 1.")
        if p.get("accumulation", 1) > 1:
            code = "MICRO_MEAN" if p.get("accumulation_mode", "exact") == "micro_mean" else "EXACT_MEMORY"
            text = "Среднее microbatch losses отличается при разных denominators." if code == "MICRO_MEAN" else "Exact сохраняет графы microbatches до backward; память растёт с accumulation."
            add("warning", code, name, text)
    if kind == "DatasetWeights":
        if p.get("gamma", .5) > 1:
            add("warning", "STRONG_COMPENSATION", name, "gamma > 1 усиливает дисбаланс сильнее inverse frequency.")
        if p.get("missing", "error") != "error":
            add("warning", "MISSING_CLASS", name, "Политика missing не создаёт положительные обучающие данные для отсутствующего класса.")
    if kind == "PresenceWeights":
        add("warning", "PRESENCE_NORMALIZATION", name, "present/empty — относительные weights; общий коэффициент однородного batch сокращается. Для абсолютного штрафа используйте empty_cost.")
    if kind == "FrequencyWeights":
        add("warning", "LEGACY_FREQUENCY", name, "Legacy epsilon скрывает zero counts; рекомендуется DatasetWeights с missing policy.")
    if kind == "LossMixer" and p.get("mode", "sum") == "normalized_sum":
        add("warning", "TERM_NORMALIZATION", name, "Общее масштабирование всех term weights сокращается.")


def _metadata(report, metadata, params):
    if not metadata:
        report.add("pending", "METADATA", "metadata", "Нужны task_mode, class_names и dataset counts для проверки источников весов и фона.")
        return
    if not isinstance(metadata, Mapping):
        report.add("error", "METADATA", "metadata", "Ожидается словарь metadata.")
        return
    add = report.add
    names = metadata.get("class_names", [])
    if names and len(set(names)) != len(names):
        add("error", "CLASS_MAPPING", "metadata", "Имена классов должны быть уникальны.")
    background = metadata.get("background_class_id")
    if background is not None and (not isinstance(background, int) or not 0 <= background < len(names)):
        add("error", "BACKGROUND", "metadata", "background_class_id вне class mapping.")
    if metadata.get("task_mode") == "multiclass" and background is None and metadata.get("contains_background"):
        add("warning", "NO_BACKGROUND", "metadata", "Softmax не предсказывает «ни один класс»: явно задайте ROI/ignore или background channel.")
    if metadata.get("split", "train") != "train":
        add("warning", "STATISTICS_SPLIT", "metadata", "Частотные веса обычно считают только по train split.")
    if metadata.get("sampling_changed"):
        add("warning", "SAMPLING_DISTRIBUTION", "metadata", "Oversampling/cropping меняет распределение; исходные inverse frequencies могут повторно компенсировать дисбаланс.")
    import torch
    for key, value in metadata.items():
        if key.endswith("_count") and isinstance(value, (list, tuple, torch.Tensor)):
            tensor = torch.as_tensor(value)
            if not torch.isfinite(tensor).all() or (tensor < 0).any():
                add("error", "COUNTS", key, "Counts должны быть конечными и неотрицательными.")
            if names and tensor.numel() != len(names):
                add("error", "COUNT_SHAPE", key, "Количество counts не совпадает с class mapping.")
    for name, p in params.items():
        if "include_classes" in p and p["include_classes"] is not None and names:
            if any(not isinstance(c, int) or not 0 <= c < len(names) for c in p["include_classes"]):
                add("error", "INCLUDE_CLASSES", name, "include_classes вне class mapping.")


def _batch(report, batch, modules, params):
    import torch
    if not isinstance(batch, Mapping):
        report.add("error", "BATCH", "batch", "Ожидается mapping context keys → values.")
        return
    for name, p in params.items():
        if ".losses." not in modules[name]["class"]:
            continue
        output = p.get("output")
        if output in batch:
            value = batch[output]
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                report.add("error", "SCALAR_LOSS", name, "Objective должен возвращать scalar tensor.")
            elif not torch.isfinite(value).all():
                report.add("error", "FINITE_LOSS", name, "Loss не является конечным.")
        else:
            report.add("pending", "LOSS_OUTPUT", name, "Loss ещё не вычислен; его shape/finite проверит первый вызов.")
        pred, target = batch.get(p.get("prediction")), batch.get(p.get("target"))
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            if pred.device != target.device:
                report.add("error", "DEVICE", name, "Prediction и target находятся на разных устройствах.")
            if not pred.is_floating_point():
                report.add("error", "DTYPE", name, "Prediction должен иметь floating dtype.")


def _runtime_references(report, name, kind, params, contract, owners):
    """Строковые runtime keys проверяются отдельно от setup-ссылок $ctx."""
    roles = set()
    if kind in {'BCEWithLogits','BinaryFocalWithLogits','FocalLoss','CrossEntropy','MulticlassFocalLoss','MSELoss','L1Loss','DiceLoss','IoULoss','TverskyLoss','OverlapLoss','GeneralizedDiceLoss','ForegroundDiceLoss'}:
        roles = {'prediction','target','positive_cost','negative_cost','label_cost','label_weight','element_weight','case_class_weight','class_weight','sample_weight','positive_weight','negative_weight','present_weight','empty_weight','empty_cost','valid_mask'}
    elif kind in {'Forward','MoveToDevice','Select','Detach','TensorTransform','AddNoise'}:
        roles = {'input'}
    elif kind == 'OptimizationManager':
        roles = {'loss','normalizer'}
    elif kind == 'PresenceWeights':
        roles = {'target','present_weight','empty_weight','valid_mask'}
    elif kind == 'TargetClassWeights':
        roles = {'target','weights'}
    readable = set().union(*(set(contract.get(k, [])) for k in ('reads','creates','updates','mutates')))
    keys = [params[k] for k in roles if isinstance(params.get(k), str) and not params[k].startswith('$')]
    if kind == 'Forward':
        inputs = params.get('inputs', [])
        keys += list(inputs.values()) if isinstance(inputs, Mapping) else list(inputs or [])
    if kind == 'WeightProduct':
        keys += list(params.get('inputs', []))
    if kind == 'LossMixer':
        for term in params.get('terms', []):
            if not isinstance(term, Mapping) or 'loss' not in term:
                report.add('error','MIXER_TERM',name,'Каждый term должен содержать loss key.')
                continue
            keys.append(term['loss'])
            if 'weight_key' in term:
                keys.append(term['weight_key'])
    for key in keys:
        if key not in owners or key not in readable:
            report.add('error','RUNTIME_KEY',name,f'{key}: нет владельца или права чтения в контракте.')
    output = params.get('output')
    if isinstance(output, str) and output not in set(contract.get('creates', [])) | set(contract.get('updates', [])):
        report.add('error','OUTPUT_KEY',name,f'{output}: нет права создавать или заменять output.')
