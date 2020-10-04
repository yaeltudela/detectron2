import torch
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
from torch import nn

from detectron2.layers import Conv2d
from detectron2.modeling.backbone.resnet import BottleneckBlock


class RefinementHead(nn.Module):

    def __init__(self, num_classes, b_blocks=1) -> None:
        super().__init__()
        self.num_classes = num_classes + 1
        self.blocks = []
        for i in range(b_blocks):
            bn = BottleneckBlock(in_channels=256, out_channels=256, bottleneck_channels=64)
            self.blocks.append(bn)
            self.add_module("bottleneck_{}".format(i + 1), bn)

        self.fc = nn.Linear(7 * 7 * 256, self.num_classes) # classes + bg

    def forward(self, x, instances):

        if self.training:
            for bn_block in self.blocks:
                x = bn_block(x)

            x = self.fc(x.view(x.size(0), -1))
            targets = torch.zeros((instances.size(0), self.num_classes), dtype=torch.float64,
                                  device=instances.device)
            targets[range(instances.size(0)), instances] = 1

            loss = sigmoid_focal_loss(x.squeeze(), targets, reduction='mean')
            # loss = F.binary_cross_entropy_with_logits(x.squeeze(), targets, reduction='mean')
            return {"loss_refine": loss}
        else:
            inst = []
            if x.numel() > 0:
                for bn_block in self.blocks:
                    x = bn_block(x)

                x = self.fc(x.view(x.size(0), -1))
                for el, instance in zip(x, instances):
                    instance = self.refine_inference_single_instance(x.squeeze(), instance)
                    inst.append(instance)
                return inst
            else:
                return instances

    def refine_inference_single_instance(self, pred_logits, instance):
        box_pred_class = instance.get('pred_classes')
        scores_pred = instance.get('scores')

        pred_class = pred_logits.max(dim=0)[1]
        if pred_class == box_pred_class:
            return instance
        else:
            if pred_class == self.num_classes:
                print("bg")
                instance.scores = instance.scores * 0.5
            else:
                # print("changes",)
                lambda_val = 0.3
                scores_pred = (1 - lambda_val) * scores_pred + lambda_val * pred_logits.softmax(dim=0)[box_pred_class]
                instance.scores = scores_pred
        return instance


class WidthHeightAreaHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=1, bias=True, norm=None, activation=None)
        self.c2 = Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=1, bias=True, norm=None, activation=None)
        self.fc2 = nn.Linear(256 * 7 * 7, 3)

    def forward(self, x, targets):
        x = self.c1(x)
        x = self.c2(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)

        if self.training:
            return {"wha_loss": (smooth_l1_loss(x, targets, beta=0., reduction='mean'))}
        else:
            return x


class ConvRegBoxHead(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.id_1 = conv1x1(256, 1024)
        self.c1 = conv3x3(256, 256)
        self.c2 = conv1x1(256, 1024)
        self.relu = nn.ReLU(inplace=True)

        self.bottlenecks = []
        for i in range(k - 1):
            bottleneck = BottleneckBlock(in_channels=1024, out_channels=1024, bottleneck_channels=256)
            self.add_module("bottleneck_{}".format(i + 1), bottleneck)
            self.bottlenecks.append(bottleneck)

        self.gab = nn.AvgPool2d((7, 7))

    def forward(self, x):
        id_1 = self.id_1(x)
        x = self.c1(x)
        x = self.c2(x)

        x = self.relu(x + id_1)

        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        x = self.gab(x)

        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding=0)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


