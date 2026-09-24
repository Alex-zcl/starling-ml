"""Schedulers, early stopping и EMA остаются обычными stateful modules."""
import torch
from ...core.module import Module


class LRScheduler(Module):
    def setup(self,optimizer,scheduler="StepLR",signal="optimizer_step",optimizer_prefix="optim.main",**kwargs):
        self.scheduler=getattr(torch.optim.lr_scheduler,scheduler)(optimizer,**kwargs)
        self.listen_signal,self.optimizer_prefix=signal,optimizer_prefix
    def reaction(self,signal,source=None,**payload):
        if signal==self.listen_signal and payload.get("optimizer",self.optimizer_prefix)==self.optimizer_prefix:
            self.scheduler.step()
    def state_dict(self):return self.scheduler.state_dict()
    def load_state_dict(self,state):self.scheduler.load_state_dict(state)


class EarlyStopping(Module):
    def setup(self,metric,mode="min",patience=5,min_delta=0.,phase="validation",metric_prefix="metrics.validation"):
        if mode not in {"min","max"} or patience<1:raise ValueError("Invalid early stopping")
        self.metric,self.mode,self.patience,self.min_delta=metric,mode,patience,min_delta
        self.phase,self.metric_prefix=phase,metric_prefix;self.best=None;self.bad=0
    def reaction(self,signal,source=None,**payload):
        if signal!="metrics_ready" or payload.get("phase")!=self.phase or payload.get("prefix")!=self.metric_prefix:return
        value=float(self.context[self.metric]);better=self.best is None or (value<self.best-self.min_delta if self.mode=="min" else value>self.best+self.min_delta)
        if better:self.best=value;self.bad=0
        else:self.bad+=1
        stop=self.bad>=self.patience
        import torch.distributed as dist
        if dist.is_initialized():
            device="cuda" if dist.get_backend()=="nccl" else "cpu"
            flag=torch.tensor(int(stop),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX);stop=bool(flag)
        if stop:self.signal("stop_requested",reason="early_stopping")
    def state_dict(self):return dict(best=self.best,bad=self.bad)
    def load_state_dict(self,state):self.best,self.bad=state["best"],state["bad"]


class EMATeacher(Module):
    def setup(self,student,teacher,decay=.99,optimizer_prefix="optim.main"):
        if not 0<=decay<1:raise ValueError("EMA decay must lie in [0,1)")
        self.student,self.teacher,self.decay,self.prefix=student,teacher,decay,optimizer_prefix
        teacher.load_state_dict(student.state_dict());teacher.requires_grad_(False);teacher.eval()
    def reaction(self,signal,source=None,**payload):
        if signal!="optimizer_step" or payload.get("optimizer")!=self.prefix:return
        with torch.no_grad():
            for t,s in zip(self.teacher.parameters(),self.student.parameters()):t.lerp_(s,1-self.decay)
            for t,s in zip(self.teacher.buffers(),self.student.buffers()):t.copy_(s)
    def state_dict(self):return self.teacher.state_dict()
    def load_state_dict(self,state):self.teacher.load_state_dict(state)
