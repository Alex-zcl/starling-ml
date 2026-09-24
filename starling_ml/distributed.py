"""DDP запускается снаружи Engine; один и тот же config работает на каждом rank."""
from contextlib import contextmanager
from datetime import timedelta
import os
import torch
from .core.module import Module


@contextmanager
def distributed_session(backend=None, timeout_seconds=120):
    """Используйте внутри torchrun; владение process group ограничено контекстом."""
    import torch.distributed as dist
    owned = not dist.is_initialized()
    if owned:
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend, timeout=timedelta(seconds=timeout_seconds))
    try:
        yield dist.get_rank()
    finally:
        if owned and dist.is_initialized():
            dist.destroy_process_group()


class DistributedModel(Module):
    """Обёртка после device placement, до optimizer; raw model остаётся владельцем weights."""
    def setup(self, model, output="model.instance", find_unused_parameters=False):
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError("Initialize distributed_session before Engine.setup")
        device = next(model.parameters()).device
        ids = [device.index] if device.type == "cuda" else None
        self.context[output] = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=ids, find_unused_parameters=find_unused_parameters,
            # Uneven validation ranks не должны синхронизировать buffers на forward.
            broadcast_buffers=False,
        )

    def state_dict(self):
        # Weights сохраняет исходный factory: дублирование DDP prefix не требуется.
        return {}


def distributed_config(config, device=None):
    """Явно адаптирует standard recipe; исходный словарь остаётся неизменным."""
    from copy import deepcopy
    result = deepcopy(config)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    result["constants"]["device"] = device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    result["modules"]["device"]["params"]["output"] = "model.placed"
    result["contracts"]["device"]["creates"] = ["model.placed"]
    result["modules"]["ddp"] = {"class": "starling_ml.distributed.DistributedModel", "params": dict(model="$ctx:model.placed", output="model.instance")}
    result["contracts"]["ddp"] = dict(reads=["model.placed"], mutates=["model.placed"], creates=["model.instance"])
    result["modules"]["batch"]["params"]["distributed"] = True
    return result
