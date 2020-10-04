# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import pycocotools.mask as mask_util

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.layers.wrappers import Max
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_rcnn_loss(pred_mask_logits, instances, maskiou_on=False, vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    mask_ratios = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if maskiou_on:
            cropped_mask = instances_per_image.gt_masks.crop(instances_per_image.proposal_boxes.tensor)
            cropped_mask = torch.tensor(
                [mask_util.area(mask_util.frPyObjects([p for p in obj], box[3]-box[1], box[2]-box[0])).sum().astype(float)
                for obj, box in zip(cropped_mask.polygons, instances_per_image.proposal_boxes.tensor)]
                )

            mask_ratios.append(
                (cropped_mask / instances_per_image.gt_masks.area())
                .to(device=pred_mask_logits.device).clamp(min=0., max=1.)
            )

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if gt_classes:
        gt_classes = cat(gt_classes, dim=0)
    else:
        gt_classes = torch.tensor([], device=pred_mask_logits.device)
    if len(gt_masks) == 0:
        if maskiou_on:
            selected_index = torch.arange(pred_mask_logits.shape[0], device=pred_mask_logits.device)
            selected_mask = pred_mask_logits[selected_index, gt_classes]
            mask_num, mask_h, mask_w = selected_mask.shape
            selected_mask = selected_mask.reshape(mask_num, 1, mask_h, mask_w)
            return pred_mask_logits.sum() * 0, selected_mask, gt_classes, None

        else:
            return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        # gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks_float = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks_float], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks_float, reduction="mean")

    if maskiou_on:
        mask_ratios = cat(mask_ratios, dim=0)
        value_eps = 1e-10 * torch.ones(gt_masks.shape[0], device=gt_classes.device, dtype=mask_ratios.dtype)
        mask_ratios = torch.max(mask_ratios, value_eps)

        pred_masks = pred_mask_logits > 0

        mask_targets_full_area = gt_masks.sum(dim=[1, 2]) / mask_ratios
        # mask_ovr = pred_masks * gt_masks
        mask_ovr_area = (pred_masks * gt_masks).sum(dim=[1, 2]).float()
        mask_union_area = pred_masks.sum(dim=[1, 2]) + mask_targets_full_area - mask_ovr_area
        value_1 = torch.ones(pred_masks.shape[0], device=gt_classes.device, dtype=mask_union_area.dtype)
        value_0 = torch.zeros(pred_masks.shape[0], device=gt_classes.device, dtype=mask_ovr_area.dtype)
        mask_union_area = torch.max(mask_union_area, value_1)
        mask_ovr_area = torch.max(mask_ovr_area, value_0)
        maskiou_targets = mask_ovr_area / mask_union_area
        # selected_index = torch.arange(pred_mask_logits.shape[0], device=gt_classes.device)
        # selected_mask = pred_mask_logits[selected_index, gt_classes]
        mask_num, mask_h, mask_w = pred_mask_logits.shape
        selected_mask = pred_mask_logits.reshape(mask_num, 1, mask_h, mask_w)
        selected_mask = selected_mask.sigmoid()

        return mask_loss, selected_mask, gt_classes, maskiou_targets.detach()
    else:
        return mask_loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])

        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, vis_period=0, maskiou_on=False, mask_attention=False):
        """
        NOTE: this interface is experimental.

        Args:
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.maskiou_on = maskiou_on
        self.mask_attention = mask_attention

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "vis_period": cfg.VIS_PERIOD,
            "maskiou_on": cfg.MODEL.MASKIOU_ON,
            "mask_attention": cfg.MODEL.ROI_MASK_HEAD.ATTENTION,
        }

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            if self.maskiou_on:
                loss, selected_mask, labels, maskiou_targets = mask_rcnn_loss(x, instances, maskiou_on=self.maskiou_on, vis_period=self.vis_period)
                if self.mask_attention:
                    return {"loss_mask": loss}, selected_mask, labels, maskiou_targets, x
                else:
                    return {"loss_mask": loss}, selected_mask, labels, maskiou_targets
            elif self.mask_attention:
                return {
                    "loss_mask": mask_rcnn_loss(x, instances, maskiou_on=self.maskiou_on, vis_period=self.vis_period)}, x
            else:
                return {"loss_mask": mask_rcnn_loss(x, instances, maskiou_on=self.maskiou_on, vis_period=self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", sam_on=False, sam_k=3, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.sam_on = sam_on

        if self.sam_on:
            self.sam = SpatialAttentionModule(kernel_size=3)

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        ret['sam_on'] = cfg.MODEL.ROI_MASK_HEAD.SAM_ON
        ret['sam_k'] = cfg.MODEL.ROI_MASK_HEAD.SAM_K
        return ret

    def layers(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)

        if self.sam_on:
            x = self.sam(x)

        x = F.relu(self.deconv(x))
        return self.predictor(x)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttentionModule, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else (1 if kernel_size == 3 else 2)

        self.conv = Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        weight_init.c2_msra_fill(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
