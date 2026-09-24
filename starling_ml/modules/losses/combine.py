"""Смешивание objectives не знает shapes, activation или task type."""
import torch
from ...core.module import Module
from ...ops.objectives import MixedObjective,mean_objective


class LossMixer(Module):
    def setup(self,terms,output="train.loss",mode="sum",eps=1e-12):
        if not terms or mode not in {"sum","normalized_sum"}:
            raise ValueError("Invalid LossMixer terms/mode")
        self.terms,self.output,self.mode=terms,output,mode

    def __call__(self):
        objectives=[]
        for term in self.terms:
            value=self.context[term["loss"]]
            if not isinstance(value,torch.Tensor) or value.numel()!=1:
                raise ValueError("LossMixer accepts scalar tensors only")
            weight=self.context[term["weight_key"]] if "weight_key" in term else term.get("weight",1.)
            weight=torch.as_tensor(weight,device=value.device,dtype=value.dtype)
            if weight.numel()!=1 or not torch.isfinite(weight):
                raise ValueError("Term weight must be a finite scalar")
            obj=getattr(value,"_starling_ml_objective",None)
            if obj is None:
                obj=mean_objective(value,torch.ones_like(value),"external_scalar_mean")
                obj.external_scalar = True
            objectives.append((weight,obj))
        self.context[self.output]=MixedObjective(tuple(objectives),self.mode).tensor()
