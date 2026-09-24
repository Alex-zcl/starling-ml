"""Стандартные конфиги — свежие Python-словари, редактируемые перед Engine."""


def get_config(task="segmentation", *, max_steps=3, seed=42, device="cpu"):
    """Маленькие synthetic recipes для проверки wiring, не pretrained pipelines."""
    if task in {"weighted_segmentation", "segmentation_validation", "gan", "detection", "instance_segmentation"}:
        from .recipes import special_config
        return special_config(task, max_steps=max_steps, seed=seed, device=device)
    available = {"segmentation", "segmentation3d", "multilabel", "classification", "regression", "language_model", "diffusion", "contrastive", "distillation", "rl_bandit"}
    if task not in available:
        raise ValueError(f"Unknown recipe {task}; choose {sorted(available)}")
    config = dict(constants=dict(max_steps=max_steps, seed=seed, device=device), modules={}, contracts={}, pipeline=[])

    def add(name, cls, params, *, creates=(), reads=(), mutates=(), updates=()):
        config["modules"][name] = dict(class_=cls, params=params)
        config["modules"][name]["class"] = config["modules"][name].pop("class_")
        config["contracts"][name] = dict(creates=list(creates), reads=list(reads), mutates=list(mutates), updates=list(updates))

    factory = "starling_ml.modules.processing.runtime.ObjectFactory"
    add("seed", "starling_ml.modules.processing.runtime.Seed", dict(seed="$const:seed"))
    dataset_task = "classification" if task == "rl_bandit" else task
    spatial = [4, 4, 4] if task == "segmentation3d" else [8, 8]
    add("dataset", factory, dict(factory="starling_ml.modules.data.general.SyntheticDataset", output="data.dataset", kwargs=dict(task=dataset_task, spatial=spatial, seed=seed)), creates=["data.dataset"])
    add("batch", "starling_ml.modules.data.general.BatchSource", dict(dataset="$ctx:data.dataset", outputs={"batch.input": "input", "batch.target": "target"}, batch_size=4, seed=seed), reads=["data.dataset"], creates=["batch.input", "batch.target"])
    architecture = "MLP"
    model_options = dict(in_features=4, out_features=3 if task in {"classification", "rl_bandit", "distillation"} else 4 if task == "contrastive" else 1)
    if task in {"segmentation", "segmentation3d", "multilabel"}:
        architecture, model_options = "Segmenter", dict(classes=3, spatial_dims=3 if task == "segmentation3d" else 2)
    elif task == "language_model":
        architecture, model_options = "CausalTokenModel", {}
    elif task == "diffusion":
        architecture, model_options = "NoisePredictor", {}
    add("model", factory, dict(factory="starling_ml.modules.models.architectures." + architecture, output="model.raw", kwargs=model_options), creates=["model.raw"])
    add("device", "starling_ml.modules.processing.runtime.PrepareModel", dict(model="$ctx:model.raw", device="$const:device", output="model.instance"), reads=["model.raw"], mutates=["model.raw"], creates=["model.instance"])
    for key in ("input", "target"):
        add("move_" + key, "starling_ml.modules.processing.runtime.MoveToDevice", dict(input="batch." + key, device="$const:device"), reads=["batch." + key], updates=["batch." + key])
    forward = dict(model="$ctx:model.instance", input="batch.input", output="model.output")
    forward_reads = ["model.instance", "batch.input"]
    body = [dict(call=n) for n in ("batch", "move_input", "move_target")]
    if task == "diffusion":
        add("noise", "starling_ml.modules.tasks.DiffusionNoise", dict(input="batch.input"), reads=["batch.input"], creates=["batch.noisy", "batch.timesteps", "batch.noise_target"])
        body.append(dict(call="noise"))
        forward.pop("input")
        forward["inputs"] = dict(sample="batch.noisy", timesteps="batch.timesteps")
        forward_reads = ["model.instance", "batch.noisy", "batch.timesteps"]
    add("forward", "starling_ml.modules.forward.Forward", forward, reads=forward_reads, mutates=["model.instance"], creates=["model.output"])
    body.append(dict(call="forward"))
    loss_cls = "CrossEntropy"
    loss = dict(prediction="model.output", target="batch.target", output="train.loss")
    reads = ["model.output", "batch.target"]
    if task == "multilabel":
        loss_cls = "BCEWithLogits"
    elif task in {"regression", "diffusion"}:
        loss_cls = "MSELoss"
        if task == "diffusion":
            loss["target"] = "batch.noise_target"
            reads = ["model.output", "batch.noise_target"]
    elif task == "language_model":
        loss["class_dim"] = -1
    if task == "distillation":
        add("teacher", factory, dict(factory="starling_ml.modules.models.architectures.MLP", output="teacher.raw", kwargs=model_options), creates=["teacher.raw"])
        add("teacher_device", "starling_ml.modules.processing.runtime.PrepareModel", dict(model="$ctx:teacher.raw", device=device, output="teacher.instance"), reads=["teacher.raw"], mutates=["teacher.raw"], creates=["teacher.instance"])
        add("teacher_forward", "starling_ml.modules.forward.Forward", dict(model="$ctx:teacher.instance", input="batch.input", output="teacher.output", training=False), reads=["teacher.instance", "batch.input"], mutates=["teacher.instance"], creates=["teacher.output"])
        add("loss", "starling_ml.modules.tasks.DistillationLoss", dict(student="model.output", teacher="teacher.output", output="train.loss"), reads=["model.output", "teacher.output"], creates=["train.loss"])
        body.append(dict(call="teacher_forward"))
    elif task == "contrastive":
        add("augment", "starling_ml.modules.processing.runtime.AddNoise", dict(input="batch.input", output="batch.second", scale=.1), reads=["batch.input"], creates=["batch.second"])
        add("second_forward", "starling_ml.modules.forward.Forward", dict(model="$ctx:model.instance", input="batch.second", output="model.second"), reads=["model.instance", "batch.second"], mutates=["model.instance"], creates=["model.second"])
        add("loss", "starling_ml.modules.tasks.ContrastiveLoss", dict(first="model.output", second="model.second", output="train.loss"), reads=["model.output", "model.second"], creates=["train.loss"])
        body += [dict(call="augment"), dict(call="second_forward")]
    elif task == "rl_bandit":
        add("action", "starling_ml.modules.tasks.CategoricalAction", dict(logits="model.output"), reads=["model.output"], creates=["rl.action", "rl.log_probability", "rl.entropy"])
        add("reward", "starling_ml.modules.tasks.BanditReward", dict(action="rl.action", target="batch.target"), reads=["rl.action", "batch.target"], creates=["rl.reward"])
        add("loss", "starling_ml.modules.tasks.PolicyLoss", dict(log_probability="rl.log_probability", advantage="rl.reward", entropy="rl.entropy"), reads=["rl.log_probability", "rl.reward", "rl.entropy"], creates=["train.loss"])
        body += [dict(call="action"), dict(call="reward")]
    else:
        add("loss", "starling_ml.modules.losses.basic." + loss_cls, loss, reads=reads, creates=["train.loss"])
    body.append(dict(call="loss"))
    if task in {"segmentation", "segmentation3d", "multilabel"}:
        # Два losses имеют собственные weights; Mixer не переносит их между terms.
        config["modules"]["loss"]["params"]["output"] = "loss.pointwise"
        config["contracts"]["loss"]["creates"] = ["loss.pointwise"]
        add("dice", "starling_ml.modules.losses.overlap.DiceLoss", dict(prediction="model.output", target="batch.target", output="loss.dice", mode="multilabel" if task == "multilabel" else "multiclass", empty_target="false_positive"), reads=["model.output", "batch.target"], creates=["loss.dice"])
        add("mix", "starling_ml.modules.losses.combine.LossMixer", dict(terms=[dict(loss="loss.pointwise"), dict(loss="loss.dice")]), reads=["loss.pointwise", "loss.dice"], creates=["train.loss"])
        body += [dict(call="dice"), dict(call="mix")]
    optim_keys = ["optim.main." + k for k in ("optimizer", "did_step", "step", "micro_step", "grad_norm", "skipped")]
    add("optimizer", "starling_ml.modules.optimization.OptimizationManager", dict(model="$ctx:model.instance", lr=.001), reads=["model.instance", "train.loss"], mutates=["model.instance"], creates=optim_keys)
    add("run", "starling_ml.modules.control.run.RunManager", dict(max_steps="$const:max_steps", validate_at_end=False), creates=["run." + k for k in ("step", "phase", "validation_index", "validation_due", "stop")])
    body += [dict(call="optimizer"), dict(when=dict(condition="$ctx:optim.main.did_step", body=[dict(signal="step_completed")]))]
    config["pipeline"] = [dict(signal="run_started"), dict(while_=None)]
    config["pipeline"][1] = {"while": dict(condition="$ctx:run.stop", equals=False, max_iterations=100000, body=body)}
    return config


STANDARD_CONFIGS = ("segmentation", "segmentation3d", "multilabel", "classification", "regression", "language_model", "diffusion", "contrastive", "distillation", "rl_bandit")

STANDARD_CONFIGS += ("weighted_segmentation", "segmentation_validation", "gan", "detection", "instance_segmentation")
