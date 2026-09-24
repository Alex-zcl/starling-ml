# Возможности и границы 0.3.0

| Область | Реализовано | Что подтверждено / что остаётся |
|---|---|---|
| Classification/regression | factories, shapes [B]/[B,C], CE/focal/MSE/L1, device | synthetic CPU training; segmentation confusion stats можно применять к multiclass classification |
| 2D/3D segmentation | BCE/CE/focal + overlap, void masks, dataset stats, training/validation | 2D/3D synthetic CPU; реальный датасет не предоставлен |
| MONAI | network/transform/sliding-window adapters с eval/no_grad | нет real MONAI volume validation в текущей среде |
| SMP | lazy model factory | mock integration test; pretrained weights не скачивались |
| NLP | causal token model, class_dim, ignore labels; HF processor/tokenizer/forward/generate | synthetic GRU training; tokenizer/padding/shift реальной модели должен задавать recipe |
| VLM / Florence | HF processor, model, forward, generate adapters | нет готового проверенного Florence fine-tuning recipe и task metrics |
| Diffusion | DDPM noise process, epsilon/sample/v targets, scheduler/model adapters | synthetic noise-prediction training; latent/text-conditioned pretrained pipeline не проверен |
| GAN | два optimizer, detach, freeze/unfreeze, non-saturating BCE | synthetic CPU recipe; advanced GAN regularization не включена |
| Detection/instance | ragged collate, query model, Hungarian matching, class/box/mask loss | synthetic training; AP при одном IoU threshold; это не COCO mAP и не полный DETR/Mask R-CNN |
| Contrastive | symmetric InfoNCE, две views, optional global negatives | local CPU; global negatives требует равных rank batches, multi-rank не проверен |
| Distillation | KL temperature, teacher detach, EMA module | synthetic CPU; task-specific teacher preprocessing остаётся в config |
| RL | categorical policy, PPO clipped objective, GAE, replay, Gymnasium boundary | contextual bandit recipe и GAE test; нет проверенного полного PPO rollout training loop |
| DDP | wrapper, sampler partition, stats all-reduce, metrics aggregation, per-rank checkpoints | код и reproducer; Gloo socket запрещён средой, GPU/NCCL не проверен |
| Независимые эксперименты | spawn launcher, отдельный cwd/config/checkpoint | два процесса CPU проверены; devices/seed задаёт config |
| Resume | model/optimizer/scaler/modules/RNG + встроенный batch cursor | точное CPU совпадение с непрерывным запуском; arbitrary multi-worker data stream/env state не гарантируется |

## Optional dependencies

Extras в pyproject: detection, progress, tracking, smp, transformers, diffusers,
monai, nnunet, surfaces, all. Базовый пакет требует только torch и PyYAML.

Wrappers с **kwargs передают параметры upstream библиотеке. Анализатор проверяет
собственную границу, но не гарантирует совместимость всех версий optional API.
Сами PyPI names/версии не бронировались и пакет не публиковался.

## Ограничения распределённого запуска

Все ranks должны использовать согласованные configs и число optimizer steps.
Train BatchSource отбрасывает хвост длиной меньше world_size и не дублирует samples.
Для validation используйте отдельный finite source без padding; collectives метрик
должны выполняться после окончания окна на каждом rank. DDP forward не синхронизирует
buffers: для BatchNorm требуется собственная явно выбранная политика.
EarlyStopping синхронизирует stop; произвольный пользовательский stop-сигнал следует
посылать согласованно на всех ranks. Console/Tqdm выводят только rank 0; сторонние
tracking wrappers требуют отдельной rank policy в config. Checkpoint paths должны
содержать {rank}; восстановление предполагает прежний world size и dataset length.

## Предоставление production recipe

Нужны реальные данные/доступ к модели, заявленная метрика, фиксированный набор
dependencies и отдельный end-to-end test. Не следует превращать таблицу adapters
в заявление о готовности произвольной задачи. Эти границы важны при передаче архива
следующему разработчику или другой модели.
