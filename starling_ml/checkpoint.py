"""Checkpoint I/O вне Engine: общий state protocol и воспроизводимый RNG."""
from pathlib import Path
import random
import torch


def save_checkpoint(engine,path):
    return save_snapshot(engine.state_dict(),path)


def save_snapshot(modules,path):
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_world_size()>1:
        raise RuntimeError("Use per-rank checkpoint paths; call save_rank_snapshot from every rank")
    return _write(modules,path)


def _write(modules,path):
    state=dict(format_version=1,modules=modules,python_rng=random.getstate(),torch_rng=torch.get_rng_state())
    if torch.cuda.is_available():state["cuda_rng"]=torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy_rng"]=np.random.get_state()
    except ImportError:pass
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    torch.save(state,temp);temp.replace(path)
    return path


def save_rank_snapshot(modules,path):
    import torch.distributed as dist
    rank=dist.get_rank() if dist.is_initialized() else 0
    if dist.is_initialized() and dist.get_world_size() > 1 and "{rank}" not in str(path):
        raise ValueError("Per-rank checkpoint path must contain {rank}")
    return _write(modules,str(path).format(rank=rank))


def load_checkpoint(engine,path):
    engine.setup()
    state=read_checkpoint(path)
    engine.load_state_dict(state["modules"])
    restore_rng(state)
    return engine


def read_checkpoint(path):
    # Full training state содержит Python RNG: загружать только доверенные локальные файлы.
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("format_version")!=1:raise ValueError("Unknown checkpoint format")
    return state


def restore_rng(state):
    random.setstate(state["python_rng"]);torch.set_rng_state(state["torch_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():torch.cuda.set_rng_state_all(state["cuda_rng"])
    if "numpy_rng" in state:
        import numpy as np
        np.random.set_state(state["numpy_rng"])
