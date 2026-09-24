"""Тонкие адаптеры pure objectives; численные значения также разрешены как веса."""
import torch.nn.functional as F
from ...core.module import Module
from ...ops.pointwise import binary_objective, categorical_loss_objective
from ...ops.reduction import pointwise_objective


def _read(context,value):
    return context[value] if isinstance(value,str) else value


class BCEWithLogits(Module):
    gamma_default=0.

    def setup(self,prediction="model.logits",target="batch.mask",output="loss.bce",
              positive_cost=None,negative_cost=None,positive_weight=None,negative_weight=None,
              element_weight=None,case_class_weight=None,class_weight=None,sample_weight=None,
              valid_mask=None,ignore_index=-100,reduction="pair_mean",
              element_normalization="weight_sum",gamma=None):
        self.prediction,self.target,self.output=prediction,target,output
        self.weights=dict(positive_cost=positive_cost if positive_cost is not None else positive_weight,
                          negative_cost=negative_cost if negative_cost is not None else negative_weight,
                          element_weight=element_weight,case_class_weight=case_class_weight,
                          class_weight=class_weight,sample_weight=sample_weight,valid_mask=valid_mask)
        self.options=dict(gamma=self.gamma_default if gamma is None else gamma,ignore_index=ignore_index,
                          reduction=reduction,element_normalization=element_normalization)

    def __call__(self):
        self.context[self.output]=binary_objective(self.context[self.prediction],self.context[self.target],
            **{k:_read(self.context,v) for k,v in self.weights.items()},**self.options).tensor()


class BinaryFocalWithLogits(BCEWithLogits):
    gamma_default=2.


class CrossEntropy(Module):
    gamma_default=0.

    def setup(self,prediction="model.logits",target="batch.target",output="loss.ce",
              label_cost=None,label_weight=None,element_weight=None,sample_weight=None,
              valid_mask=None,ignore_index=-100,label_smoothing=0.,class_dim=1,
              reduction="element_mean",element_normalization="weight_sum",gamma=None):
        self.prediction,self.target,self.output=prediction,target,output
        self.weights=dict(label_cost=label_cost if label_cost is not None else label_weight,
                          element_weight=element_weight,sample_weight=sample_weight,valid_mask=valid_mask)
        self.options=dict(gamma=self.gamma_default if gamma is None else gamma,
                          ignore_index=ignore_index,label_smoothing=label_smoothing,class_dim=class_dim,
                          reduction=reduction,element_normalization=element_normalization)

    def __call__(self):
        self.context[self.output]=categorical_loss_objective(self.context[self.prediction],self.context[self.target],
            **{k:_read(self.context,v) for k,v in self.weights.items()},**self.options).tensor()


class MulticlassFocalLoss(CrossEntropy):
    gamma_default=2.


class _RegressionLoss(Module):
    loss_function=None

    def setup(self,prediction="model.output",target="batch.target",output="loss.regression",
              element_weight=None,case_class_weight=None,class_weight=None,sample_weight=None,
              valid_mask=None,reduction="pair_mean",element_normalization="weight_sum"):
        self.prediction,self.target,self.output=prediction,target,output
        self.weights=dict(element_weight=element_weight,case_class_weight=case_class_weight,
                          class_weight=class_weight,sample_weight=sample_weight,valid_mask=valid_mask)
        self.options=dict(reduction=reduction,element_normalization=element_normalization)

    def __call__(self):
        prediction=self.context[self.prediction]
        target=self.context[self.target].to(prediction.dtype)
        if prediction.shape!=target.shape:
            raise ValueError("Regression prediction and target shapes must match; broadcasting is ambiguous")
        from ...ops.objectives import stable
        from ...ops.reduction import element_map
        prediction, target = stable(prediction), stable(target)
        mask = _read(self.context, self.weights["valid_mask"])
        if mask is not None:
            valid = element_map(mask, prediction).bool()
            prediction = prediction.where(valid, 0.)
            target = target.where(valid, 0.)
        loss=self.loss_function(prediction,target,reduction="none")
        self.context[self.output]=pointwise_objective(loss,**self.options,
            **{k:_read(self.context,v) for k,v in self.weights.items()}).tensor()


class MSELoss(_RegressionLoss):
    loss_function=staticmethod(F.mse_loss)


class L1Loss(_RegressionLoss):
    loss_function=staticmethod(F.l1_loss)


FocalLoss=BinaryFocalWithLogits
