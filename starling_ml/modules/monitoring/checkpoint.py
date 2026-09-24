"""Сохранение полного состояния; загрузка выполняется после всех setup."""
from ...core.module import Module


class Checkpoint(Module):
    def setup(self,path,model=None,optimizer=None,step_key="run.step",signal="run_end",full_state=True):
        self.path,self.model,self.optimizer=path,model,optimizer
        self.step_key,self.listen_signal,self.full_state=step_key,signal,full_state

    def reaction(self,signal,source=None,**payload):
        if signal!=self.listen_signal:return
        import torch
        import torch.distributed as dist
        from pathlib import Path
        step=int(self.context.get(self.step_key,0));rank=dist.get_rank() if dist.is_initialized() else 0
        if dist.is_initialized() and dist.get_world_size()>1 and "{rank}" not in self.path:
            raise ValueError("Distributed checkpoint path must contain {rank} for per-rank RNG/data state")
        path=self.path.format(step=step,rank=rank)
        if self.full_state:
            from ...checkpoint import save_rank_snapshot
            save_rank_snapshot(self._snapshot(),self.path.replace("{step}",str(step)))
        else:
            if self.model is None:raise ValueError("Weights-only checkpoint requires model")
            state=dict(model=self.model.state_dict(),step=step)
            if self.optimizer is not None:state["optimizer"]=self.optimizer.state_dict()
            Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(state,path)


class LoadCheckpoint(Module):
    def setup(self,path):self.path=path
    def ready(self):
        import torch.distributed as dist
        from ...checkpoint import read_checkpoint,restore_rng
        rank=dist.get_rank() if dist.is_initialized() else 0
        state=read_checkpoint(self.path.format(rank=rank))
        # Loader может отсутствовать в сохранённом config; остальные модули обязаны совпасть.
        expected=set(self._snapshot())-{self.name}
        if set(state["modules"])-{self.name} != expected:
            raise ValueError("Checkpoint module names differ")
        self._restore(state["modules"],strict=False);restore_rng(state)
