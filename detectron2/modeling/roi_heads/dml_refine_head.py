import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, get_norm


class RefinementHeadMetric(nn.Module):

    def __init__(self, num_classes, embedding_size=128) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.c1 = Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
            bias=not "",
            norm=get_norm("", 256),
            activation=F.relu, )
        self.c3 = Conv2d(
            256,
            64,
            kernel_size=1,
            bias=not "",
            norm=get_norm("", 128),
            activation=F.relu, )
        self.fc = nn.Linear(7 * 7 * 64, embedding_size)

    def forward(self, x, instances):

        if self.training:
            x = self.c1(x)
            x = self.c3(x)
            x = self.fc(x.view(x.size(0), -1))
            return x
        else:
            if x.numel() > 0:
                x = self.c1(x)
                x = self.c3(x)
                x = self.fc(x.view(x.size(0), -1))

                return x, instances
            else:
                return None, instances


def recompute_proto_centroids(proto_centroids, embeddings, targets):
    with torch.no_grad():
        batch_clusters = torch.zeros_like(proto_centroids)
        for k in range(proto_centroids.size(0)):
            if embeddings[targets == k].numel() > 0:
                batch_clusters[k] = embeddings[targets == k].mean(dim=0)
                proto_centroids[k] = torch.stack([proto_centroids[k], batch_clusters[k]], dim=0).mean(dim=0)
        return proto_centroids

        # for k in range(proto_centroids.size(0)):
        #     if embeddings[targets == k].numel() > 0:
        #         proto_centroids[k] = torch.cat([embeddings[targets == k], proto_centroids[k][None, ...]]).mean(dim=0)
        # return proto_centroids