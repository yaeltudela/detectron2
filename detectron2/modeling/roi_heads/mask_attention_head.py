from torch import nn

from detectron2.layers import Conv2d


class MaskAttention(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        # self.c1 = Conv2d(num_classes, 256, kernel_size=(1, 1),  bias=True, norm=None, activation=None)
        self.c2 = Conv2d(num_classes, 256, kernel_size=(1, 1), stride=2,  bias=True, norm=None, activation=None)
        self.c3 = Conv2d(256, 256, kernel_size=(1, 1), stride=2, bias=True, norm=None, activation=None)
        # self.max_pool = nn.MaxPool2d(4, 4)

    def forward(self, mask_output, box_pooler_features):
        if self.training:
            # x = self.c1(mask_output)
            x = mask_output
            x = self.c2(x)
            x = self.c3(x)
            # x = self.max_pool(x)

            x = x * box_pooler_features

            return x
        else:
            return box_pooler_features