import torch
from torch import nn
from detectron2.structures.boxes import pairwise_iou


class GIOULoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        for inp, out in zip(input,target):
            ious = pairwise_iou(inp, out)



