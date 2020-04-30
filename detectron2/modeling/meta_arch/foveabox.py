# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List

import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss, sigmoid_focal_loss
from torch import nn

from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
# from .build import META_ARCH_REGISTRY

__all__ = ["FoveaBox"]


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


# @META_ARCH_REGISTRY.register()
class FoveaBox(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.FOVEABOX.NUM_CLASSES
        self.in_features = cfg.MODEL.FOVEABOX.IN_FEATURES
        self.edge_list = cfg.MODEL.FOVEABOX.EDGE_LIST
        self.scale_ranges = cfg.MODEL.FOVEABOX.SCALE_RANGES
        self.strides = cfg.MODEL.FOVEABOX.STRIDES
        self.sigma = cfg.MODEL.FOVEABOX.SIGMA

        self.focal_loss_alpha = cfg.MODEL.FOVEABOX.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FOVEABOX.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.FOVEABOX.SMOOTH_L1_LOSS_BETA

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        self.head = FoveaBoxHead(cfg, feature_shapes)

        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        # TODO Head Fovea
        cls_logits, pred_boxes = self.head(features)

        feature_map_sizes = [cls_logit.size()[-2:] for cls_logit in cls_logits]
        points = []
        for feature_map_size in feature_map_sizes:
            x_range = torch.arange(feature_map_size[1]) + 0.5
            y_range = torch.arange(feature_map_size[0]) + 0.5
            y, x = torch.meshgrid(y_range, x_range)
            points.append((x, y))

        gt_bboxes_list, gt_classes_list = self.method_name(feature_map_sizes, gt_instances, points)

        print(gt_bboxes_list[0][0].shape, gt_classes_list[0][0].shape, pred_boxes[0].shape, cls_logits[0].shape)
        flatten_labels = [torch.cat([labels_level_img.flatten().unsqueeze(-1).expand(-1, self.num_classes) for labels_level_img in labels_level]) for labels_level in zip(*gt_classes_list)]
        flatten_labels = torch.cat(flatten_labels).float()

        flatten_bbox_targets = [torch.cat([bbox_targets_level_img.reshape(-1, 4) for bbox_targets_level_img in bbox_targets_level]) for bbox_targets_level in zip(*gt_bboxes_list)]
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)

        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for cls_score in cls_logits]
        flatten_cls_scores = torch.cat(flatten_cls_scores)

        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in pred_boxes]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)

        num_imgs = sum([len(instance) for instance in gt_instances])
        pos_inds = (flatten_labels > 0).nonzero().view(-1)
        num_pos = len(pos_inds)

        # Continue like retina
        if self.training:
            return self.losses(flatten_labels, flatten_bbox_targets, flatten_cls_scores, flatten_bbox_preds, num_pos, num_imgs, pos_inds)
        else:
            #results = self.inference(cls_logits, pred_boxes, anchors, images.image_sizes)
            results = None
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # r = detector_postprocess(results_per_image, height, width)
                # processed_results.append({"instances": r})
            return processed_results

    @torch.no_grad()
    def method_name(self, feature_map_sizes, gt_instances, points):
        gt_classes_list = []
        gt_bboxes_list = []
        for instance in gt_instances:
            gt_bboxes_raw = instance.gt_boxes.tensor
            gt_labels_raw = instance.gt_classes.view(-1, 1)

            gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                    gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
            label_list = []
            bbox_target_list = []
            for base_len, (lower_bound, upper_bound), stride, featmap_size, (y, x) \
                    in zip(self.edge_list, self.scale_ranges, self.strides, feature_map_sizes, points):
                labels = gt_labels_raw.new_zeros(featmap_size)
                bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1], 4) + 1

                hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
                if len(hit_indices) == 0:
                    label_list.append(labels)
                    bbox_target_list.append(torch.log(bbox_targets))
                    continue

                _, hit_index_order = torch.sort(-gt_areas[hit_indices])
                hit_indices = hit_indices[hit_index_order]
                gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
                gt_labels = gt_labels_raw[hit_indices]
                half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
                half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
                pos_left = torch.ceil(gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long(). \
                    clamp(0, featmap_size[1] - 1)
                pos_right = torch.floor(gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long(). \
                    clamp(0, featmap_size[1] - 1)
                pos_top = torch.ceil(gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long(). \
                    clamp(0, featmap_size[0] - 1)
                pos_down = torch.floor(gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long(). \
                    clamp(0, featmap_size[0] - 1)
                for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                        zip(pos_left, pos_top, pos_right, pos_down, gt_labels,
                            gt_bboxes_raw[hit_indices, :]):
                    labels[py1:py2 + 1, px1:px2 + 1] = label
                    bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = (stride * x[py1:py2 + 1,
                                                                          px1:px2 + 1] - gt_x1) / base_len
                    bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = (stride * y[py1:py2 + 1,
                                                                          px1:px2 + 1] - gt_y1) / base_len
                    bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = (gt_x2 - stride * x[py1:py2 + 1,
                                                                                  px1:px2 + 1]) / base_len
                    bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = (gt_y2 - stride * y[py1:py2 + 1,
                                                                                  px1:px2 + 1]) / base_len
                bbox_targets = bbox_targets.clamp(min=1. / 16, max=16.)
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))

            gt_classes_list.append(label_list)
            gt_bboxes_list.append(bbox_target_list)
        return gt_bboxes_list, gt_classes_list

    def losses(self, gt_classes_target, gt_boxes, pred_class_logits, pred_boxes, num_pos, num_ims, pos_inds):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """

        loss_cls = sigmoid_focal_loss(
            pred_class_logits,
            gt_classes_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_pos + num_ims)

        # regression loss

        if num_pos > 0:
            pred_boxes = pred_boxes[pos_inds]
            gt_boxes = gt_boxes[pos_inds]
            pos_weights = gt_boxes.new_zeros(gt_boxes.size()) + 1.0
            loss_box_reg = smooth_l1_loss(
                pred_boxes,
                gt_boxes,
                beta=self.smooth_l1_loss_beta,
                reduction="mean",
            ) / max(1, num_pos)
        else:
            loss_box_reg = torch.tensor([0], dtype=pred_boxes.dtype, device=pred_boxes.device)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    @torch.no_grad()
    def get_ground_truth(self):
        pass

class FoveaBoxHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        num_classes = cfg.MODEL.FOVEABOX.NUM_CLASSES
        in_channels = input_shape[0].channels
        feature_channels = 256
        num_convs = cfg.MODEL.FOVEABOX.NUM_CONVS

        self.box_branch = None
        self.cls_branch = None

        cls_subnet = []
        bbox_subnet = []
        for i in range(num_convs):
            cls_subnet.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU()
                )
            )
            bbox_subnet.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU()
                )
            )

        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.bbox_pred = nn.Conv2d(feature_channels, 4, kernel_size=3, stride=1, padding=1)

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_score = nn.Conv2d(feature_channels, num_classes, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        torch.nn.init.normal_(self.cls_score.bias, std=0.01)

    def forward(self, features):

        cls_logits = []
        bbox_reg = []
        for feature in features:
            cls_logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))

        return cls_logits, bbox_reg


if __name__ == '__main__':
    import cv2
    import numpy as np

    cfg = get_cfg()
    cfg.merge_from_file("../../../configs/fovea/Test.yaml")

    batch = torch.load("/home/devsodin/batch.pth")
    model = FoveaBox(cfg)
    print(model)
    model = model.cuda()
    a = model(batch)
    # model([{"image": torch.tensor(a)}])
    print(a)
