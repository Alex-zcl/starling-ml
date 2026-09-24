"""Маленькие обычные torch.nn модели для воспроизводимых recipes."""
import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self,in_features=4,out_features=1,hidden=16):
        super().__init__(nn.Linear(in_features,hidden),nn.ReLU(),nn.Linear(hidden,out_features))


class Segmenter(nn.Module):
    def __init__(self,in_channels=1,classes=3,spatial_dims=2,hidden=8):
        super().__init__();conv=nn.Conv3d if spatial_dims==3 else nn.Conv2d
        self.net=nn.Sequential(conv(in_channels,hidden,3,padding=1),nn.ReLU(),conv(hidden,classes,1))
    def forward(self,x):return self.net(x)


class CausalTokenModel(nn.Module):
    """GRU causal baseline; HF-модели подключаются существующим HFForward."""
    def __init__(self,vocabulary=16,hidden=16):
        super().__init__();self.embedding=nn.Embedding(vocabulary,hidden)
        self.gru=nn.GRU(hidden,hidden,batch_first=True);self.head=nn.Linear(hidden,vocabulary)
    def forward(self,input_ids):return self.head(self.gru(self.embedding(input_ids))[0])


class NoisePredictor(nn.Module):
    def __init__(self,features=4,hidden=16):
        super().__init__();self.net=MLP(features+1,features,hidden)
    def forward(self,sample,timesteps):return self.net(torch.cat([sample,timesteps.float().reshape(-1,1)/1000],1))


class QueryDetector(nn.Module):
    """Небольшой set-prediction baseline для ragged boxes и instance masks."""
    def __init__(self, features=4, classes=2, queries=3, masks=False):
        super().__init__()
        self.queries, self.classes, self.masks = queries, classes, masks
        self.head = nn.Linear(features, queries * (classes + 1 + 4 + (16 if masks else 0)))

    def forward(self, x):
        values = self.head(x).reshape(len(x), self.queries, -1)
        points = values[..., self.classes + 1:self.classes + 5].sigmoid()
        boxes = torch.cat((torch.minimum(points[..., :2], points[..., 2:]), torch.maximum(points[..., :2], points[..., 2:])), -1)
        result = []
        for i in range(len(x)):
            record = dict(logits=values[i, :, :self.classes + 1], boxes=boxes[i])
            if self.masks:
                record['masks'] = values[i, :, self.classes + 5:].reshape(self.queries, 4, 4)
            result.append(record)
        return result
