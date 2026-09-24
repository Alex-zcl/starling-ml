"""Оптимизация с явной политикой accumulation и сохраняемым состоянием."""
import torch
from ..core.module import Module
from ..ops.objectives import mean_objective


class OptimizationManager(Module):
    def setup(self,model,loss="train.loss",optimizer="adam",lr=1e-3,weight_decay=0.,momentum=.9,
              accumulation=1,accumulation_mode="exact",normalizer=None,amp=False,clip_grad_norm=None,
              fail_on_nonfinite=True,fail_on_missing_grad=True,prefix="optim.main",
              partial="error",distributed=True,parameter_groups=None,**optimizer_kwargs):
        self.model,self.loss,self.prefix=model,loss,prefix
        self.accumulation=int(accumulation)
        if self.accumulation<1 or accumulation_mode not in {"exact","micro_mean"} or partial not in {"error","drop","flush"}:
            raise ValueError("Invalid accumulation configuration")
        self.accumulation_mode,self.normalizer,self.partial=accumulation_mode,normalizer,partial
        self.distributed=distributed;self.amp=bool(amp and torch.cuda.is_available())
        self.clip_grad_norm,self.fail_on_nonfinite=clip_grad_norm,fail_on_nonfinite
        self.fail_on_missing_grad=fail_on_missing_grad
        if hasattr(optimizer,"step") and not isinstance(optimizer,type):
            self.optimizer=optimizer
        else:
            cls=self._optimizer_class(optimizer)
            kwargs=dict(lr=lr,weight_decay=weight_decay,**optimizer_kwargs)
            if cls is torch.optim.SGD:kwargs["momentum"]=momentum
            parameters=model.parameters()
            if parameter_groups is not None:
                # Параметр принадлежит ровно одной группе; regex не используется скрыто.
                named=dict(model.named_parameters());used=set();parameters=[]
                for spec in parameter_groups:
                    names=spec["names"]
                    if used.intersection(names):raise ValueError("Duplicate optimizer parameter")
                    used.update(names)
                    parameters.append(dict(params=[named[n] for n in names],**{k:v for k,v in spec.items() if k!="names"}))
            self.optimizer=cls(parameters,**kwargs)
        self.scaler=torch.amp.GradScaler("cuda",enabled=self.amp)
        self.micro_step=0;self.optimizer_step=0;self.pending_count=0;self.pending=None
        self.optimizer.zero_grad(set_to_none=True)
        for name,value in dict(optimizer=self.optimizer,did_step=False,step=0,micro_step=0,grad_norm=0.,skipped=False).items():
            self.context[f"{prefix}.{name}"]=value

    @staticmethod
    def _optimizer_class(name):
        if callable(name) and not isinstance(name,str):return name
        classes={"adam":torch.optim.Adam,"adamw":torch.optim.AdamW,"sgd":torch.optim.SGD}
        if name.lower() not in classes:raise ValueError(f"Unknown optimizer: {name}")
        return classes[name.lower()]

    def _parameters(self):
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def __call__(self):
        self.context[f"{self.prefix}.did_step"]=False
        self.context[f"{self.prefix}.skipped"]=False
        loss=self.context[self.loss]
        if not isinstance(loss,torch.Tensor) or loss.numel()!=1:
            raise ValueError("Optimization requires scalar tensor loss")
        if not loss.requires_grad:
            raise RuntimeError("Loss is detached from optimizer parameters")
        if self.fail_on_nonfinite and not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite loss")
        self.micro_step+=1;self.pending_count+=1
        self.context[f"{self.prefix}.micro_step"]=self.micro_step
        if self.accumulation_mode=="exact":
            objective=getattr(loss,"_starling_ml_objective",None)
            if objective is None:
                if self.normalizer is not None:
                    den=torch.as_tensor(self.context[self.normalizer],device=loss.device,dtype=loss.dtype)
                    objective=mean_objective(loss*den,den,"explicit_external_mean")
                else:
                    import torch.distributed as dist
                    if self.accumulation>1 or (self.distributed and dist.is_initialized()):
                        raise ValueError("Exact accumulation/DDP of external scalar requires normalizer key or native Objective")
                    objective=mean_objective(loss,torch.ones_like(loss),"single_scalar")
            def has_external(obj):
                return getattr(obj, "external_scalar", False) or any(has_external(t) for _, t in getattr(obj, "terms", ()))
            import torch.distributed as dist
            if has_external(objective) and (self.accumulation > 1 or (self.distributed and dist.is_initialized())):
                raise ValueError("Mixed external scalar has no exact denominator; use native Objective")
            self.pending=objective if self.pending is None else self.pending.merge(objective)
        else:
            self.scaler.scale(loss/self.accumulation).backward()
        if self.pending_count==self.accumulation:self._step()

    def _step(self):
        if not self.pending_count:return
        if self.accumulation_mode=="exact":
            loss=self.pending.value(distributed=self.distributed)
            self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        parameters=self._parameters()
        norms=[p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        if not norms and self.fail_on_missing_grad:
            raise RuntimeError("Loss has no gradients for optimizer parameters")
        norm=torch.stack(norms).norm() if norms else torch.tensor(0.)
        self.context[f"{self.prefix}.grad_norm"]=float(norm.cpu())
        finite=bool(torch.isfinite(norm))
        if not finite and self.fail_on_nonfinite:
            raise FloatingPointError("Non-finite gradient norm")
        skipped=not finite
        if finite:
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters,float(self.clip_grad_norm))
            old_scale=self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skipped=self.scaler.get_scale()<old_scale
        elif self.amp:
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending=None;self.pending_count=0
        self.context[f"{self.prefix}.skipped"]=skipped
        self.context[f"{self.prefix}.did_step"]=not skipped
        if not skipped:
            self.optimizer_step+=1
            self.context[f"{self.prefix}.step"]=self.optimizer_step
            self.signal("optimizer_step",optimizer=self.prefix,step=self.optimizer_step)

    def reaction(self,signal,source=None,**payload):
        if signal=="flush_optimization" and payload.get("optimizer",self.prefix)==self.prefix:
            if self.pending_count:
                if self.partial=="error":raise RuntimeError("Incomplete accumulation window")
                if self.partial=="drop":
                    self.pending=None;self.pending_count=0;self.optimizer.zero_grad(set_to_none=True)
                    self.context[f"{self.prefix}.did_step"]=False
                else:
                    if self.accumulation_mode=="micro_mean":
                        for p in self._parameters():
                            if p.grad is not None:p.grad.mul_(self.accumulation/self.pending_count)
                    self._step()
        if signal=="run_end" and self.pending_count:
            raise RuntimeError("Flush/drop incomplete accumulation before run_end")

    def state_dict(self):
        if self.pending_count:
            raise RuntimeError("Checkpoint only at optimizer boundary; flush or drop pending accumulation first")
        return dict(optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),
                    micro_step=self.micro_step,optimizer_step=self.optimizer_step)

    def load_state_dict(self,state):
        self.optimizer.load_state_dict(state["optimizer"]);self.scaler.load_state_dict(state["scaler"])
        self.micro_step=state["micro_step"];self.optimizer_step=state["optimizer_step"]
        self.pending=None;self.pending_count=0
        self.context[f"{self.prefix}.micro_step"]=self.micro_step
        self.context[f"{self.prefix}.step"]=self.optimizer_step
        self.context[f"{self.prefix}.did_step"]=False


class Adam(OptimizationManager):
    """Legacy convenience adapter; полноценный путь использует OptimizationManager."""
    def setup(self,model,lr=1e-3,loss="train.loss",weight_decay=0.):
        # Старый контракт имел один ключ optimizer.instance, поэтому оставляем компактный путь.
        self.loss=loss;self.optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
        self.context["optimizer.instance"]=self.optimizer

    def __call__(self):
        self.optimizer.zero_grad(set_to_none=True);self.context[self.loss].backward();self.optimizer.step()

    def reaction(self,*args,**kwargs):pass
    def state_dict(self):return {"optimizer":self.optimizer.state_dict()}
    def load_state_dict(self,state):self.optimizer.load_state_dict(state["optimizer"])
