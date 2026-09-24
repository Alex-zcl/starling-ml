# Отчёт проверки 0.3.0

Дата: 2026-09-24. Проверялось фактическое содержимое релиза Starling ML 0.3.0.

## Среда

Linux, Python 3.12.14, PyTorch 2.14.0+cpu, PyYAML 6.0.3, SciPy 1.17.0.
CUDA недоступна. Network sockets для Gloo запрещены средой.
Заявленный Python >=3.10 / torch>=2.3 — границы package metadata, а не проверенная
матрица всех версий; перед публикацией нужен CI.

## Выполнено

| Проверка | Результат |
|---|---|
| unittest discovery | **72 tests, OK** |
| 15 standard configs | обучение на CPU с реальными forward/backward/optimizer steps |
| classification/segmentation/token shapes | отдельные math tests и synthetic recipes |
| FP16/BF16 zero weights | finite loss и backward на CPU |
| большие FP16 overlap sums | finite forward/backward |
| ignore masks, unknown targets | исключены из соответствующих denominators и confusion stats |
| focal costs | проверено значение по ручной формуле; p берётся до weighting |
| present/empty vs empty_cost | проверено сокращение importance и сохранение абсолютного cost |
| class_macro / pair_mean / sample_mean / batch_stats | численные invariants и merge gradients |
| exact accumulation | loss/gradient equivalence с единым batch, включая unequal valid counts |
| checkpoint resume | параметры совпадают с непрерывным CPU запуском без допуска |
| config analyzer | syntax/signature/references/cycles, warnings, metadata, published loss outputs |
| weight signals | результат не зависит от порядка producer/consumer в config |
| независимые workers | два spawn-процесса, разные outputs/checkpoints |
| wheel build/install | wheel собран, установлен в отдельный target, импорт подтверждён из installed path |
| installed public API | все 15 recipes запущены из установленного wheel |
| CLI analyzer | опубликована понятная сводка weighted segmentation |
| source distribution | собран setuptools backend |
| release metadata | distribution `starling-ml`, import `starling_ml`, MIT license и URLs проверены внутри wheel |
| Twine validation | wheel и sdist: PASSED |

Logs: validation/tests.txt, validation/wheel.txt, validation/analyzer.txt.
Количество unit tests не включает число subtests: 15 recipes выполняются в одном
parameterized unittest method. Несколько проверок optional integrations используют
mock factories; они не доказывают реальную загрузку pretrained models.

## Заблокировано или не проверялось

**Двухпроцессный DDP не прошёл до этапа обучения.** `examples/check_ddp.py` попытался
создать Gloo process group с file rendezvous и завершился при создании TCP socket:
`Operation not permitted`. Это ограничение среды; корректность полного DDP пути
не подтверждена. Log находится в validation/ddp.txt. Автоматическое повышение прав
не использовалось. Script оставлен для повторения на обычной машине.

Не выполнялись CUDA/AMP/NCCL, реальные pretrained SMP/HF/Florence/Diffusers/MONAI
pipelines, внешние tracking services, обучение на реальных данных и benchmark качества.
PPO primitives/replay/env adapter не являются полным проверенным online trainer.
Матрица ограничений подробно приведена в INTEGRATIONS.md.

## Что именно означает exact

Проверено объединение математических stats native objectives. Сравнение целого batch
с microbatches использует детерминированные models без BatchNorm/dropout. Оно не
доказывает идентичность stochastic forward или расширение contrastive negatives между
microbatches. Exact mode хранит graphs и не заявляется как способ снизить память.

## Воспроизводимые команды

```bash
python -m pip install -e '.[detection]'
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python examples/check_launcher.py
python examples/check_ddp.py
python -m starling_ml --recipe weighted_segmentation --analyze-only
```

Ранее сообщавшиеся «78 тестов» не подтверждались сохранёнными файлами; данный отчёт
заменяет эту цифру фактическим повторным прогоном. При будущих изменениях обновлять
отчёт после тестов, а не сохранять старую цифру автоматически.
