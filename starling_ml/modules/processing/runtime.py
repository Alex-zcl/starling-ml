"""Небольшие runtime-операции над объектами и градиентами."""
import importlib
import torch
from ...core.module import Module


def tree_to(value,device,dtype=None):
    if isinstance(value,torch.Tensor):return value.to(device=device,dtype=dtype if value.is_floating_point() else None)
    if isinstance(value,dict):return {k:tree_to(v,device,dtype) for k,v in value.items()}
    if isinstance(value,list):return [tree_to(v,device,dtype) for v in value]
    if isinstance(value,tuple):return tuple(tree_to(v,device,dtype) for v in value)
    return value


class MoveToDevice(Module):
    def setup(self,input,output=None,device="cpu",dtype=None):
        self.input,self.output,self.device=input,output or input,device
        self.dtype=getattr(torch,dtype) if dtype else None
    def __call__(self):self.context[self.output]=tree_to(self.context[self.input],self.device,self.dtype)


class ModelToDevice(Module):
    def setup(self,model,device="cpu",dtype=None):
        model.to(device=device,dtype=getattr(torch,dtype) if dtype else None)


class ObjectFactory(Module):
    """Один прозрачный factory для пользовательских моделей/datasets/optimizers."""
    def setup(self,factory,output,args=None,kwargs=None):
        module,name=factory.rsplit(".",1)
        self.context[output]=getattr(importlib.import_module(module),name)(*(args or []),**(kwargs or {}))


class Select(Module):
    def setup(self,input,output,path):self.input,self.output,self.path=input,output,path
    def __call__(self):
        value=self.context[self.input]
        for key in self.path:value=value[key] if isinstance(value,(dict,list,tuple)) else getattr(value,key)
        self.context[self.output]=value


class Detach(Module):
    def setup(self,input,output):self.input,self.output=input,output
    def __call__(self):self.context[self.output]=self.context[self.input].detach()


class ModelMode(Module):
    def setup(self,model,training=None,requires_grad=None):
        self.model,self.training,self.requires_grad=model,training,requires_grad
    def __call__(self):
        if self.training is not None:self.model.train(self.training)
        if self.requires_grad is not None:
            for p in self.model.parameters():p.requires_grad_(self.requires_grad)


class ZeroGrad(Module):
    def setup(self,optimizer):self.optimizer=optimizer
    def __call__(self):self.optimizer.zero_grad(set_to_none=True)


class Backward(Module):
    def setup(self,loss,retain_graph=False):self.loss,self.retain_graph=loss,retain_graph
    def __call__(self):self.context[self.loss].backward(retain_graph=self.retain_graph)


class OptimizerStep(Module):
    def setup(self,optimizer):self.optimizer=optimizer
    def __call__(self):self.optimizer.step()


class TensorTransform(Module):
    """Явные layout/shape-преобразования для tokens, features и mask channels."""
    def setup(self,input,output,operation,**options):self.input,self.output,self.operation,self.options=input,output,operation,options
    def __call__(self):
        x=self.context[self.input];p=self.options
        if self.operation=="movedim":x=x.movedim(p["source"],p["destination"])
        elif self.operation=="reshape":x=x.reshape(*p["shape"])
        elif self.operation=="unsqueeze":x=x.unsqueeze(p["dim"])
        elif self.operation=="squeeze":x=x.squeeze(p.get("dim",1))
        else:raise ValueError("Unknown tensor transform")
        self.context[self.output]=x


class Seed(Module):
    """Один seed устанавливается до setup factories; RNG сохраняется checkpoint."""
    def setup(self, seed=42):
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass


class PrepareModel(Module):
    """Выходная ссылка гарантирует device placement до создания optimizer."""
    def setup(self, model, output="model.instance", device="cpu"):
        self.context[output] = model.to(device)


class AddNoise(Module):
    """Простая аугментация второй view в contrastive smoke recipe."""
    def setup(self, input, output, scale=.1):
        self.input, self.output, self.scale = input, output, scale

    def __call__(self):
        value = self.context[self.input]
        self.context[self.output] = value + torch.randn_like(value) * self.scale
