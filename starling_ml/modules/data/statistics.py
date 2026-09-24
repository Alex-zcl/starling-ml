"""Статистика полного iterable dataset; отдельно от обучающего iterator."""
from ...core.module import Module
from ...ops.statistics import DatasetStatistics


class DatasetStats(Module):
    def setup(self,dataset,class_names,mode="multilabel",target="target",study_id="study_id",
              valid_mask="valid_mask",presence_known="presence_known",output="data.statistics",
              split="train",fingerprint=None,transform_scope="source",ignore_index=-100,sample_unit="study",study_complete="study_complete"):
        import torch.distributed as dist
        result=None
        if not dist.is_initialized() or dist.get_rank()==0:
            stats=DatasetStatistics(class_names,mode,ignore_index,split,fingerprint,transform_scope,sample_unit)
            for index,record in enumerate(dataset):
                stats.update(record[target],record.get(study_id,index),record.get(valid_mask),record.get(presence_known),record.get(study_complete))
            result=stats.result()
        if dist.is_initialized():
            values=[result];dist.broadcast_object_list(values,src=0);result=values[0]
        self.context[output]=result
