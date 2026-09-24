"""BCE/focal/CE — чистая математика без Context."""
import torch
import torch.nn.functional as F
from .objectives import stable, positive_weight
from .reduction import pointwise_objective, categorical_objective, element_map
from .tensors import channel_view


def binary_objective(logits,target,gamma=0.,positive_cost=None,negative_cost=None,
                     valid_mask=None,ignore_index=-100,**reduction):
    logits=stable(logits)
    if target.shape != logits.shape:
        raise ValueError("Binary target and logits must have identical shapes")
    valid = target != ignore_index
    if valid_mask is not None:
        valid = valid & element_map(valid_mask,logits).bool()
    logits = torch.where(valid, logits, torch.zeros_like(logits))
    target=torch.where(valid,target,torch.zeros_like(target)).to(logits.dtype)
    if ((target<0)|(target>1)).any() or gamma<0:
        raise ValueError("Binary targets must be probabilities; gamma >= 0")
    pos=channel_view(positive_weight(1. if positive_cost is None else positive_cost,logits),logits)
    neg=channel_view(positive_weight(1. if negative_cost is None else negative_cost,logits),logits)
    # Компонентная формула корректна и для soft targets, без смешанных cross-terms.
    lp=F.softplus(-logits);ln=F.softplus(logits)
    if gamma:
        lp=lp*torch.sigmoid(-logits).pow(gamma)
        ln=ln*torch.sigmoid(logits).pow(gamma)
    loss=pos*target*lp+neg*(1-target)*ln
    return pointwise_objective(loss,valid_mask=valid,**reduction)


def categorical_loss_objective(logits,target,gamma=0.,label_cost=None,
        valid_mask=None,ignore_index=-100,label_smoothing=0.,class_dim=1,**reduction):
    if target.shape == logits.shape:
        target = target.movedim(class_dim, 1)
    logits=stable(logits).movedim(class_dim,1)
    if gamma<0 or not 0<=label_smoothing<1:
        raise ValueError("Invalid focal gamma or label smoothing")
    classes=logits.shape[1]
    if target.ndim == logits.ndim and target.shape[1]==1:
        target=target.squeeze(1)
    if target.shape == logits.shape:
        soft=target.to(logits.dtype)
        if (soft<0).any() or not torch.allclose(soft.sum(1),torch.ones_like(soft.sum(1)),atol=1e-5):
            raise ValueError("Soft categorical targets must sum to one")
        valid=torch.ones_like(soft[:,0],dtype=torch.bool)
    else:
        if target.dtype.is_floating_point and not torch.equal(target,target.round()):
            raise ValueError("Hard categorical targets must be integer labels")
        valid=target!=ignore_index
        if valid_mask is not None:
            valid=valid & element_map(valid_mask,target.float(),channel=False).bool()
        safe=torch.where(valid,target,torch.zeros_like(target)).long()
        if ((safe<0)|(safe>=classes)).any():
            raise ValueError("Categorical target outside class mapping")
        soft=F.one_hot(safe,classes).movedim(-1,1).to(logits.dtype)
    if valid_mask is not None:
        valid=valid & element_map(valid_mask,valid.float(),channel=False).bool()
    soft=soft*(1-label_smoothing)+label_smoothing/classes
    logp=F.log_softmax(logits,dim=1)
    base=-logp
    if gamma:
        # Вероятности вычисляются ДО class costs, включая soft-target extension.
        base=base*(1-logp.exp()).pow(gamma)
    if label_cost is not None:
        base=base*channel_view(positive_weight(label_cost,logits),logits)
    return categorical_objective((soft*base).sum(1),valid,**reduction)
