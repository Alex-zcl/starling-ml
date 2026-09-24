"""Независимые overlap adapters с одинаковой явной системой весов."""
import torch
from ...core.module import Module
from ...ops.losses import overlap_objective,generalized_dice_objective,overlap_from_probabilities
from .basic import _read


class OverlapLoss(Module):
    kind="dice"

    def setup(self,prediction="model.logits",target="batch.mask",output="loss.overlap",
              mode="multilabel",activation=None,aggregation="sample",reduction="pair_mean",
              element_weight=None,case_class_weight=None,class_weight=None,sample_weight=None,
              positive_weight=None,negative_weight=None,empty_cost=1.,valid_mask=None,
              empty_target="ignore",smooth=1.,alpha=.5,beta=.5,ignore_index=-100,include_classes=None,
              present_weight=None,empty_weight=None):
        self.prediction,self.target,self.output=prediction,target,output
        self.options=dict(mode=mode,activation=activation or ("softmax" if mode=="multiclass" else "sigmoid"),
            aggregation=aggregation,reduction=reduction,empty_target=empty_target,smooth=smooth,
            alpha=alpha,beta=beta,ignore_index=ignore_index,include_classes=include_classes)
        self.weights=dict(element_weight=element_weight,case_class_weight=case_class_weight,
            class_weight=class_weight,sample_weight=sample_weight,positive_weight=positive_weight,
            negative_weight=negative_weight,empty_cost=empty_cost,valid_mask=valid_mask,
            present_weight=present_weight,empty_weight=empty_weight)

    def __call__(self):
        self.context[self.output]=overlap_objective(self.context[self.prediction],self.context[self.target],
            kind=self.kind,**self.options,**{k:_read(self.context,v) for k,v in self.weights.items()})


class DiceLoss(OverlapLoss):
    kind="dice"


class IoULoss(OverlapLoss):
    kind="iou"


class TverskyLoss(OverlapLoss):
    kind="tversky"


class GeneralizedDiceLoss(Module):
    def setup(self,prediction="model.logits",target="batch.mask",output="loss.generalized_dice",
              mode="multiclass",activation=None,element_weight=None,class_weight=None,
              sample_weight=None,case_class_weight=None,valid_mask=None,ignore_index=-100,
              empty_target="ignore",smooth=1.,include_classes=None):
        self.prediction,self.target,self.output=prediction,target,output
        self.options=dict(mode=mode,activation=activation or ("softmax" if mode=="multiclass" else "sigmoid"),
            ignore_index=ignore_index,empty_target=empty_target,smooth=smooth,include_classes=include_classes)
        self.weights=dict(element_weight=element_weight,class_weight=class_weight,sample_weight=sample_weight,
                          case_class_weight=case_class_weight,valid_mask=valid_mask)

    def __call__(self):
        self.context[self.output]=generalized_dice_objective(self.context[self.prediction],self.context[self.target],
            **self.options,**{k:_read(self.context,v) for k,v in self.weights.items()})


class ForegroundDiceLoss(Module):
    def setup(self,prediction="model.logits",target="batch.target",output="loss.foreground_dice",
              background=0,aggregation="sample",element_weight=None,case_class_weight=None,
              sample_weight=None,valid_mask=None,ignore_index=-100,empty_target="ignore",smooth=1.):
        self.prediction,self.target,self.output=prediction,target,output
        self.background,self.ignore_index=background,ignore_index
        self.options=dict(aggregation=aggregation,empty_target=empty_target,smooth=smooth)
        self.weights=dict(element_weight=element_weight,case_class_weight=case_class_weight,sample_weight=sample_weight)
        self.valid_mask=valid_mask

    def __call__(self):
        logits=self.context[self.prediction]
        target=self.context[self.target]
        if target.ndim==logits.ndim:
            target=target.squeeze(1)
        valid=target!=self.ignore_index
        if self.valid_mask is not None:
            valid=valid & _read(self.context,self.valid_mask).bool()
        prediction=1-logits.float().softmax(1)[:,self.background:self.background+1]
        foreground=((target!=self.background)&valid).unsqueeze(1).to(prediction.dtype)
        self.context[self.output]=overlap_from_probabilities(prediction,foreground,valid_mask=valid.unsqueeze(1),
            **self.options,**{k:_read(self.context,v) for k,v in self.weights.items()})
