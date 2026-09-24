# Передача проекта для интеграции в pip

## Задача следующего разработчика

Сохранить существующий публичный API и математический смысл losses; интегрировать
пакет в выбранную инфраструктуру публикации/CI. Пользователь ожидает короткий импорт,
стандартный config с небольшими изменениями и возможность показать анализ до запуска.

```python
from starling_ml import Engine, get_config, analyze_config
config = get_config("segmentation")
config["modules"]["optimizer"]["params"]["lr"] = 1e-4
print(analyze_config(config))
Engine(config).run()
```

Distribution name: `starling-ml`; import name: `starling_ml`; версия: `0.3.0`.
Проект лицензирован по MIT и предназначен для публичного репозитория
`Alex-zcl/starling-ml`. Имя distribution `starling-ml` проверено как свободное в
PyPI 24 сентября 2026 года. Import name намеренно не сокращается до `starling`:
это пространство уже используется отдельным пакетом.

## Что находится в архиве

- исходники Python-пакета;
- pyproject.toml, MANIFEST.in, собранные wheel и source distribution в dist;
- unit/regression tests и отдельные scripts для DDP и независимых workers;
- 15 standard configs внутри пакета, старые YAML, examples;
- документация архитектуры, весов, анализатора, migration, состояние проверок;
- исторические audit/design материалы;
- SHA256SUMS.txt для проверки содержимого, кроме самого файла checksums.

Torch/dependencies, datasets, model checkpoints, caches и временные файлы разработки
в архив не включаются. Реальные личные данные не используются.

## Воспроизведение после распаковки

```bash
python -m venv .venv
# активировать venv обычным для своей ОС способом
python -m pip install -e '.[detection]'
python -m unittest discover -s tests -v
starling-ml --recipe weighted_segmentation --analyze-only
python examples/quickstart.py
python examples/check_launcher.py
python examples/check_ddp.py
```

Последняя команда требует среду, где Gloo разрешено открывать сокеты. GPU-тесты
нужно добавить отдельно; отсутствие GPU нельзя оформлять как успешный AMP/NCCL тест.

Для пересборки:

```bash
python -m pip install build
python -m build
```

В текущей среде build frontend не установлен; сборка выполнена прямым вызовом
setuptools.build_meta.build_wheel/build_sdist, затем wheel установлен отдельно от source
и из него запущен публичный сценарий. Итог — в VALIDATION_REPORT.md.

## Что нельзя «упрощать» при упаковке

- Не заменять sum(N)/sum(D) на среднее локальных средних.
- Не смешивать pixel costs, outer importance и empty cost.
- Не восстанавливать focal probability из weighted CE.
- Не удалять background pixels при исключении background class term Dice.
- Не объявлять unknown target отрицательным.
- Не скрывать exact accumulation memory cost или DDP/GPU статус.
- Не добавлять task-specific режимы в Engine; используйте ops и modules.
- Не превращать экспериментальные adapters в обещание production поддержки.

## Рекомендуемые следующие шаги

1. CI на поддерживаемых Python/PyTorch версиях, CPU и доступной CUDA.
2. Проверка двух ranks Gloo, NCCL, uneven valid counts, checkpoint на каждом rank,
   accumulation>1 с реальными моделями; отдельный контроль BatchNorm.
3. Реальные dataset recipes для SMP/MONAI/HF/VLM/diffusion, полноценный RL rollout.
4. Проверка Trusted Publisher, GitHub Release и установки опубликованного wheel.
5. Форматирование нового кода единым formatter без удаления русских объяснений.

Нельзя считать пункт выполненным только потому, что здесь существует wrapper или
примитив. Точная матрица готовности находится в INTEGRATIONS.md.
