"""Map-style datasets и сохраняемый batch cursor для прозрачного resume."""
import torch
from torch.utils.data import default_collate
from ...core.module import Module


class BatchSource(Module):
    def setup(self,dataset,outputs,batch_size=4,shuffle=True,seed=42,cycle=True,
              drop_last=False,reset_on=None,distributed=False,collate_fn=None):
        self.dataset,self.outputs=dataset,outputs
        if isinstance(collate_fn, str):
            import importlib
            path, name = collate_fn.rsplit(".", 1)
            collate_fn = getattr(importlib.import_module(path), name)
        self.collate_fn = collate_fn or default_collate
        self.batch_size=int(batch_size);self.shuffle=shuffle;self.cycle=cycle;self.drop_last=drop_last
        if self.batch_size<1 or not len(dataset):raise ValueError("Nonempty dataset and positive batch_size required")
        self.generator=torch.Generator().manual_seed(seed)
        self.reset_on=set(reset_on or []);self.epoch=0;self.cursor=0;self.distributed=distributed
        self.rank=0;self.world=1
        if distributed:
            import torch.distributed as dist
            if not dist.is_initialized():raise RuntimeError("Distributed BatchSource needs initialized process group")
            self.rank,self.world=dist.get_rank(),dist.get_world_size()
        self._new_order()

    def _new_order(self):
        n=len(self.dataset)
        order=torch.randperm(n,generator=self.generator).tolist() if self.shuffle else list(range(n))
        if self.distributed:
            # Не дублируем samples ради padding: train ranks получают одинаковую длину.
            if self.cycle:order=order[:len(order)//self.world*self.world]
            order=order[self.rank::self.world]
        self.order=order;self.cursor=0
        if not order or (self.drop_last and len(order)<self.batch_size):
            raise ValueError("Rank has no usable batch")

    def __call__(self):
        end=min(self.cursor+self.batch_size,len(self.order))
        if self.cursor>=len(self.order) or (self.drop_last and end-self.cursor<self.batch_size):
            if not self.cycle:raise StopIteration
            self.epoch+=1;self._new_order();end=min(self.batch_size,len(self.order))
        records=[self.dataset[i] for i in self.order[self.cursor:end]];self.cursor=end
        # Ragged boxes/tokens требуют custom collate: объект callable можно передать dataset wrapper-ом.
        batch=self.collate_fn(records)
        for output,key in self.outputs.items():self.context[output]=batch[key]

    def reaction(self,signal,source=None,**payload):
        if signal in self.reset_on:self._new_order()

    def state_dict(self):
        return dict(order=self.order,cursor=self.cursor,epoch=self.epoch,rng=self.generator.get_state(),
                    dataset_length=len(self.dataset),rank=self.rank,world=self.world)

    def load_state_dict(self,state):
        if (state["dataset_length"],state["rank"],state["world"])!=(len(self.dataset),self.rank,self.world):
            raise ValueError("Resume requires same dataset length/rank/world size")
        self.order=state["order"];self.cursor=state["cursor"];self.epoch=state["epoch"]
        self.generator.set_state(state["rng"])


class SyntheticDataset:
    """Конечный deterministic dataset только для executable recipes и тестов."""
    def __init__(self,task="segmentation",size=32,classes=3,seed=42,features=4,spatial=(8,8),sequence_length=8,vocabulary=16):
        g=torch.Generator().manual_seed(seed)
        self.records=[]
        for i in range(size):
            if task in {"segmentation","multilabel","segmentation3d"}:
                x=torch.randn((1,*spatial),generator=g)
                if task=="multilabel":y=torch.stack([(x[0]>-.5+c*.5).float() for c in range(classes)])
                else:y=((x[0]+2)*classes/4).long().clamp(0,classes-1)
            elif task=="classification":
                x=torch.randn(features,generator=g);y=(x.sum()>0).long() % classes
            elif task=="regression":
                x=torch.randn(features,generator=g);y=x.sum().reshape(1)
            elif task=="language_model":
                x=torch.randint(vocabulary,(sequence_length,),generator=g);y=torch.roll(x,-1)
            elif task in {"diffusion","contrastive","distillation","gan"}:
                x=torch.randn(features,generator=g);y=x.clone()
            else:raise ValueError(f"Unknown synthetic task {task}")
            self.records.append(dict(input=x,target=y,study_id=str(i)))
    def __len__(self):return len(self.records)
    def __getitem__(self,index):return self.records[index]


class DetectionDataset:
    """Разное число объектов, включая пустые изображения, проверяет ragged collate."""
    def __init__(self, size=12, seed=42, masks=False):
        generator = torch.Generator().manual_seed(seed)
        self.records = []
        for i in range(size):
            count = i % 3
            target = dict(labels=torch.arange(count) % 2, boxes=torch.tensor([[.1, .1, .4, .5], [.5, .4, .9, .9]])[:count])
            if masks:
                target['masks'] = torch.zeros(count, 4, 4)
                target['masks'][:, 1:3, 1:3] = 1
            self.records.append(dict(input=torch.randn(4, generator=generator), target=target))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def detection_collate(records):
    """Только input имеет общую форму; targets остаются списком отдельных объектов."""
    return dict(input=torch.stack([r['input'] for r in records]), target=[r['target'] for r in records])
