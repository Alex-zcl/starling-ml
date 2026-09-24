"""Проверяемые counts dataset: study, sample и element не подменяют друг друга."""
import torch


class DatasetStatistics:
    def __init__(self,class_names,mode="multilabel",ignore_index=-100,split="train",fingerprint=None,transform_scope="source",sample_unit="study"):
        if sample_unit not in {"study", "patch", "sample"}:
            raise ValueError("sample_unit must be study, patch or sample")
        self.sample_unit = sample_unit
        self.names=list(class_names);self.mode=mode;self.ignore_index=ignore_index
        self.meta=dict(split=split,dataset_fingerprint=fingerprint,transform_scope=transform_scope,
                       completeness="exact",sample_unit=sample_unit,class_names=self.names,task_mode=mode)
        self.positive=torch.zeros(len(self.names),dtype=torch.int64)
        self.negative=torch.zeros_like(self.positive)
        self.sample_positive=torch.zeros_like(self.positive);self.sample_negative=torch.zeros_like(self.positive)
        self.studies={};self.samples=0;self.elements=0

    def update(self,target,study_id,valid_mask=None,presence_known=None,study_complete=None):
        target=torch.as_tensor(target).detach().cpu()
        classes=len(self.names)
        if self.mode=="multiclass":
            valid=target!=self.ignore_index
            if target.is_floating_point() and not torch.equal(target[valid], target[valid].round()):
                raise ValueError("Counts require integer class labels")
            safe=torch.where(valid,target,torch.zeros_like(target)).long()
            if ((safe<0)|(safe>=classes)).any():
                raise ValueError("Statistics target outside class mapping")
            y=torch.nn.functional.one_hot(safe,classes).movedim(-1,0).reshape(classes,-1)
            valid=valid.reshape(1,-1).expand_as(y)
        elif self.mode in {"multilabel","binary"}:
            if target.numel()%classes:
                raise ValueError("Statistics expects [C,...] target per sample")
            y=target.reshape(classes,-1)
            valid=y!=self.ignore_index
            if ((y[valid]!=0)&(y[valid]!=1)).any():
                raise ValueError("Pixel counts require hard binary targets; soft mass is a different statistic")
        else:
            raise ValueError("Unknown statistics mode")
        if valid_mask is not None:
            mask=torch.as_tensor(valid_mask).cpu().bool()
            if mask.numel()==y.shape[1]:mask=mask.reshape(1,-1)
            else:mask=mask.reshape(classes,-1)
            valid=valid & mask.expand_as(y)
        positive=((y==1)&valid).sum(1);negative=((y==0)&valid).sum(1)
        present=positive>0
        complete=valid.all(1)
        if presence_known is not None:
            complete=torch.as_tensor(presence_known).bool().expand(classes)
        self.positive+=positive;self.negative+=negative
        self.sample_positive+=present;self.sample_negative+=(~present)&complete
        previous=self.studies.get(str(study_id))
        # Один study может содержать несколько samples; positive учитывается один раз.
        study_known = complete if self.sample_unit == "study" else torch.zeros_like(complete)
        if study_complete is not None:
            study_known = complete & torch.as_tensor(study_complete).bool().expand(classes)
        self.studies[str(study_id)]=(present if previous is None else previous[0]|present,
                                     study_known if previous is None else previous[1]|study_known)
        self.samples+=1;self.elements+=y.shape[1]

    def result(self):
        classes=len(self.names)
        sp=torch.zeros(classes,dtype=torch.int64);sn=torch.zeros_like(sp)
        for present,complete in self.studies.values():
            sp+=present;sn+=(~present)&complete
        return dict(self.meta,study_count=len(self.studies),sample_count=self.samples,total_element_count=self.elements,
            positive_element_count=self.positive.clone(),negative_element_count=self.negative.clone(),
            valid_element_count=self.positive+self.negative,positive_study_count=sp,negative_study_count=sn,
            known_presence_study_count=sp+sn,unknown_presence_study_count=len(self.studies)-sp-sn,
            positive_sample_count=self.sample_positive.clone(),negative_sample_count=self.sample_negative.clone(),
            unknown_presence_sample_count=self.samples-self.sample_positive-self.sample_negative)


def frequency_factors(counts,target_fraction=None,gamma=.5,normalization="expectation",bounds=None,missing="error"):
    """q/f с явной политикой отсутствующего класса и фиксированной dataset-нормировкой."""
    counts=torch.as_tensor(counts,dtype=torch.float64)
    if (counts<0).any() or not torch.isfinite(counts).all() or gamma<0:
        raise ValueError("Invalid counts/gamma")
    absent=counts==0
    if absent.any() and missing=="error":
        raise ValueError("Zero-count group: choose explicit missing='zero' or 'neutral'")
    if missing not in {"error","zero","neutral"}:
        raise ValueError("Unknown missing policy")
    if counts.numel() == 0 or counts.ndim != 1:
        raise ValueError("Counts must be a nonempty vector")
    if absent.all():
        return (torch.ones_like(counts) if missing == "neutral" else torch.zeros_like(counts)).float()
    f=counts/counts.sum().clamp_min(1)
    q=torch.full_like(f,1/f.numel()) if target_fraction is None else torch.as_tensor(target_fraction,dtype=f.dtype)
    if q.shape!=f.shape or (q<0).any() or not torch.isfinite(q).all() or q.sum()<=0:
        raise ValueError("target_fraction must be a positive distribution with matching shape")
    q=q/q.sum()
    raw=(q/f.clamp_min(torch.finfo(f.dtype).tiny)).pow(gamma)
    raw=torch.where(absent,torch.ones_like(raw) if missing=="neutral" else torch.zeros_like(raw),raw)
    if normalization=="expectation":raw=raw/(f*raw).sum().clamp_min(1e-12)
    elif normalization=="mean":raw=raw/raw.mean().clamp_min(1e-12)
    elif normalization!="none":raise ValueError("Unknown normalization")
    if bounds is not None:raw=raw.clamp(min=bounds[0],max=bounds[1])
    return raw.float()


def binary_frequency_factors(positive,negative,positive_fraction=.5,**kwargs):
    positive=torch.as_tensor(positive);negative=torch.as_tensor(negative)
    results=[frequency_factors(torch.stack([p,n]),[positive_fraction,1-positive_fraction],**kwargs)
             for p,n in zip(positive.flatten(),negative.flatten())]
    values=torch.stack(results)
    return values[:,0],values[:,1]
