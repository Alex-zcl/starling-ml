"""Detection/instance adapters с ragged targets."""
from ..core.module import Module
from ..ops.detection import detection_objective,detection_ap


class DetectionLoss(Module):
    def setup(self,prediction,target,output="train.loss",no_object_cost=.1,box_cost=5.,mask_cost=1.):
        self.prediction,self.target,self.output=prediction,target,output
        self.options=dict(no_object_cost=no_object_cost,box_cost=box_cost,mask_cost=mask_cost)
    def __call__(self):self.context[self.output]=detection_objective(self.context[self.prediction],self.context[self.target],**self.options).tensor()


class DetectionMetrics(Module):
    def setup(self,prediction,target,num_classes,prefix="metrics.validation",iou_threshold=.5,use_masks=False):
        self.prediction,self.target,self.num_classes=prediction,target,num_classes
        self.prefix,self.threshold,self.use_masks=prefix,iou_threshold,use_masks;self.predictions=[];self.targets=[]
    def __call__(self):
        for destination,key in ((self.predictions,self.prediction),(self.targets,self.target)):
            destination.extend([{k:v.detach().cpu() for k,v in item.items()} for item in self.context[key]])
    def reaction(self,signal,source=None,**payload):
        if signal=="validation_start":self.predictions=[];self.targets=[]
        if signal=="validation_end":
            predictions,targets=self.predictions,self.targets
            import torch.distributed as dist
            if dist.is_initialized():
                gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,(predictions,targets))
                predictions=[p for ps,ts in gathered for p in ps];targets=[t for ps,ts in gathered for t in ts]
            result=detection_ap(predictions,targets,self.num_classes,self.threshold,self.use_masks)
            self.context[self.prefix+".map"]=result["map"]
            self.signal("metrics_ready",phase="validation",prefix=self.prefix)
