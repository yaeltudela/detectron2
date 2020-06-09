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


class RefinementHead(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()

        self.c1 = Conv2d(
            3,
            128,
            kernel_size=3,
            padding=1,
            bias=not "",
            norm=get_norm("", 256),
            activation=F.relu, )
        self.c3 = Conv2d(
            128,
            num_classes + 1,
            kernel_size=1,
            bias=not "",
            norm=get_norm("", 128),
            activation=F.relu, )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, instances, targets=None):
        if self.training:
            x = self.c1(x)
            x = self.c3(x)
            x = self.avg(x)
            return {"loss_refine": F.cross_entropy(x.squeeze(), targets.long()) / x.size(0)}
        else:
            inst = []
            for el, instance in zip(x, instances):
                if el.numel() > 0:
                    x = self.c1(el)
                    x = self.c3(x)
                    x = self.avg(x)
                    instance = refine_inference_single_instance(x.squeeze(), instance)
                inst.append(instance)
            return inst


if __name__ == '__main__':
    import cv2
    from kornia.utils import image_to_tensor

    a = RefinementHead(2).cuda()
    a.training = False

    b = cv2.resize(cv2.imread("/home/devsodin/cropped.jpg"), (96, 96))

    from detectron2.structures import Instances

    b = image_to_tensor(b).float().cuda()
    print(a(b[None, ...], None))
