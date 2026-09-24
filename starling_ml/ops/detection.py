"""Box/instance matching и metrics; optional scipy импортируется только для matching."""
import torch
import torch.nn.functional as F
from .objectives import mean_objective


def box_iou(a,b):
    size=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp_min(0)
    intersection=size.prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return intersection/(area_a[:,None]+area_b[None,:]-intersection).clamp_min(1e-8)


def detection_objective(predictions,targets,no_object_cost=.1,box_cost=5.,mask_cost=1.):
    """Set prediction: Hungarian matching, background CE, matched box/mask losses.

    Последний logits channel означает no-object. Boxes должны быть xyxy в [0,1].
    Это определённый baseline objective, а не скрытая копия DETR/Mask R-CNN.
    """
    from scipy.optimize import linear_sum_assignment
    losses=[]
    for prediction,target in zip(predictions,targets):
        logits,boxes=prediction["logits"].float(),prediction["boxes"].float()
        labels=target["labels"].to(logits.device).long();truth=target["boxes"].to(boxes.device).float()
        if len(labels)>len(boxes):raise ValueError("More target objects than prediction queries")
        class_target=torch.full((len(boxes),),logits.shape[-1]-1,device=logits.device,dtype=torch.long)
        loss=boxes.sum()*0
        if len(labels):
            cost=-logits.softmax(-1)[:,labels]+box_cost*torch.cdist(boxes,truth,p=1)+(1-box_iou(boxes,truth))
            rows,cols=linear_sum_assignment(cost.detach().cpu().numpy())
            rows=torch.as_tensor(rows,device=boxes.device);cols=torch.as_tensor(cols,device=boxes.device)
            class_target[rows]=labels[cols]
            loss=loss+box_cost*F.l1_loss(boxes[rows],truth[cols])
            if "masks" in prediction and "masks" in target:
                loss=loss+mask_cost*F.binary_cross_entropy_with_logits(prediction["masks"][rows],target["masks"].to(logits.device)[cols].float())
        weights=logits.new_ones(logits.shape[-1]);weights[-1]=no_object_cost
        loss=loss+F.cross_entropy(logits,class_target,weight=weights,reduction="none").mean()
        losses.append(loss)
    if not losses:raise ValueError("Empty detection batch")
    values=torch.stack(losses)
    return mean_objective(values.sum(),values.new_tensor(len(values)),("detection",no_object_cost,box_cost,mask_cost))


def detection_ap(predictions,targets,num_classes,iou_threshold=.5,use_masks=False):
    """Dataset AP при одном IoU threshold; это НЕ полный COCO mAP."""
    result=[]
    for c in range(num_classes):
        total=sum(int((t["labels"]==c).sum()) for t in targets)
        if total==0:result.append(float("nan"));continue
        candidates=[];truths={};used={}
        for i,(p,t) in enumerate(zip(predictions,targets)):
            selected=t["labels"]==c;truths[i]=(t["masks"] if use_masks else t["boxes"])[selected]
            used[i]=set()
            for j in torch.where(p["labels"]==c)[0].tolist():candidates.append((float(p["scores"][j]),i,j))
        candidates.sort(reverse=True);tp=[]
        for score,i,j in candidates:
            p=predictions[i];truth=truths[i]
            if not len(truth):tp.append(0.);continue
            if use_masks:
                a=p["masks"][j].bool().flatten();b=truth.bool().flatten(1)
                ious=(a&b).sum(1)/(a|b).sum(1).clamp_min(1)
            else:ious=box_iou(p["boxes"][j:j+1],truth)[0]
            # Уже matched GT исключается до выбора лучшего remaining match.
            for previous in used[i]:ious[previous]=-1
            value,index=ious.max(0);matched=float(value)>=iou_threshold
            tp.append(float(matched))
            if matched:used[i].add(int(index))
        if not tp:result.append(0.);continue
        tp=torch.tensor(tp).cumsum(0);recall=tp/total;precision=tp/torch.arange(1,len(tp)+1)
        precision=torch.flip(torch.cummax(torch.flip(precision,[0]),0).values,[0])
        result.append(float(((recall-torch.cat([recall.new_zeros(1),recall[:-1]]))*precision).sum()))
    values=torch.tensor(result)
    return dict(ap=values,map=float(values.nanmean()))
