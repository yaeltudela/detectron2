import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, get_norm


def refine_inference_single_instance(score, instance):
    box_pred_class = instance.get('pred_classes')
    scores_pred = instance.get('scores')

    # print("------------")
    # print(box_pred_class)
    # print(box_pred_class.shape)
    # print(scores_pred)
    # print(scores_pred.shape)
    # print(score)
    # print(F.softmax(score, dim=0))
    # print("------------")

    scores_pred *= F.softmax(score, dim=0)[box_pred_class]

    instance.scores = scores_pred
    return instance

def metric_inference_single_instance(score, instance):
    box_pred_class = instance.get('pred_classes')
    scores_pred = instance.get('scores')

    # print("------------")
    # print(box_pred_class)
    # print(box_pred_class.shape)
    # print(scores_pred)
    # print(scores_pred.shape)
    # print(score)
    # print(F.softmax(score, dim=0))
    # print("------------")

    scores_pred *= F.softmax(score, dim=0)[box_pred_class]

    instance.scores = scores_pred
    return instance

class RefinementHead(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()

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
            num_classes + 1,
            kernel_size=1,
            bias=not "",
            norm=get_norm("", 128),
            activation=F.relu, )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, instances):
        if self.training:
            x = self.c1(x)
            x = self.c3(x)
            print(x.shape)

            x = self.avg(x)
            return {"loss_refine": F.cross_entropy(x.squeeze(), instances.long()) / x.size(0)}
        else:
            inst = []
            if x.numel() > 0:
                x = self.c1(x)
                x = self.c3(x)
                x = self.avg(x)
                for el, instance in zip(x, instances):
                    instance = refine_inference_single_instance(x.squeeze(), instance)
                    inst.append(instance)
                return inst
            else:
                return instances


class RefinementHeadMetric(nn.Module):

    def __init__(self, num_classes, embedding_size=128) -> None:
        super().__init__()

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
            num_classes + 1,
            kernel_size=3,
            padding=1,
            bias=not "",
            norm=get_norm("", 128),
            activation=F.relu, )
        self.gab = nn.AdaptiveAvgPool2d(32)
        self.fc = nn.Linear(self.gab.output_size ** 2 * (num_classes + 1), embedding_size)

    def forward(self, x, instances):

        if self.training:
            x = self.c1(x)
            x = self.c3(x)
            x = self.gab(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            if x.numel() > 0:
                x = self.c1(x)
                x = self.c3(x)
                x = self.gab(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)

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

