# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from fvcore.nn import smooth_l1_loss
from pytorch_metric_learning import losses, miners
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.maskiou_head import build_maskiou_head, mask_iou_loss, mask_iou_inference
from detectron2.modeling.roi_heads.refinement_head import RefinementHead, WidthHeightAreaHead, ConvRegBoxHead
from detectron2.modeling.roi_heads.mask_attention_head import MaskAttention
from detectron2.modeling.roi_heads.dml_refine_head import RefinementHeadMetric, recompute_proto_centroids
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, BoxMode
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, SplittedFastRCNNOutputLayers
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head, SpatialAttentionModule

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection, as_tuple=True)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to
    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_sample_fraction,
        proposal_matcher,
        proposal_append_gt=True
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of classes. Used to label background proposals.
            batch_size_per_image (int): number of proposals to use for training
            positive_sample_fraction (float): fraction of positive (foreground) proposals
                to use for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_sample_fraction = positive_sample_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_sample_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    @configurable
    def __init__(
            self,
            *,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_head: nn.Module,
            box_predictor: nn.Module,
            mask_in_features: Optional[List[str]] = None,
            mask_pooler: Optional[ROIPooler] = None,
            mask_head: Optional[nn.Module] = None,
            keypoint_in_features: Optional[List[str]] = None,
            keypoint_pooler: Optional[ROIPooler] = None,
            keypoint_head: Optional[nn.Module] = None,
            train_on_pred_boxes: bool = False,
            maskiou_on: bool = False,
            maskiou_head:Optional[nn.Module] = None,
            maskiou_weight:float = 0.,
            use_giou: bool = False,
            loss_weights: tuple = (1, 1, 1),

            refine_cls: bool = False,
            refine_cls_type:str = "",
            cls_head_bn_blocks: int = 1,
            dml_embedding_size: int = 64,

            mask_attention: bool = False,
            split_cls_head: Optional[nn.Module] = None,
            split_reg_head: Optional[nn.Module] = None,
            wha_head_on=None,

            roi_box_head_type = "",

            **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        self.maskiou_on = maskiou_on
        if self.maskiou_on:
            self.maskiou_head = maskiou_head
            self.maskiou_weight = maskiou_weight
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.use_giou = use_giou
        print("GIOU / weights: {}".format(loss_weights) if use_giou else "")
        self.smooth_l1_loss_weight = loss_weights[0]
        self.cross_entropy_loss_weight = loss_weights[1]
        self.giou_weight = loss_weights[2] if self.use_giou else 0

        self.wha_head_on = wha_head_on
        if self.wha_head_on:
            self.wha_head = WidthHeightAreaHead()

        self.pooler_attention_on = False
        if self.pooler_attention_on:
            self.pooler_box_attention = SpatialAttentionModule()

        self.refine_cls = refine_cls
        self.refine_cls_type = refine_cls_type
        if self.refine_cls:
            if refine_cls_type == "conv":
                self.refine_head = RefinementHead(self.num_classes, b_blocks=cls_head_bn_blocks)
            elif refine_cls_type == 'dml':
                self.dml_embedding_size = dml_embedding_size
                self.refine_head = RefinementHeadMetric(self.num_classes, dml_embedding_size)
                self.refine_miner = miners.TripletMarginMiner(margin=1, type_of_triplets='hard')
                self.metric_loss = losses.TripletMarginLoss(margin=0.5)

                load_centroids = True
                self.proto_centroids = torch.load("centroids.pkl") if load_centroids else torch.zeros(
                    (self.num_classes + 1, dml_embedding_size), device='cuda')
                if self.proto_centroids.size(1) != dml_embedding_size:
                    self.proto_centroids = torch.zeros((self.num_classes + 1, dml_embedding_size), device='cuda')
                print("DML head added")

            print("Refine score head added")

        self.mask_attention = mask_attention
        if self.mask_attention:
            self.attention_layers = MaskAttention(self.num_classes)
        self.split_reg_head = split_reg_head
        self.split_cls_head = split_cls_head

        self.roi_box_head_type = roi_box_head_type

    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def reweight_losses(self, losses):
        if "loss_box_reg" in losses.keys():
            losses.update({"loss_box_reg": losses["loss_box_reg"] * self.smooth_l1_loss_weight})
        if "loss_cls" in losses.keys():
            losses.update({"loss_cls": losses["loss_cls"] * self.cross_entropy_loss_weight})
        if "loss_giou" in losses.keys():
            losses.update({"loss_giou": losses["loss_giou"] * self.giou_weight})
        if "loss_split_reg" in losses.keys():
            losses.update({"loss_split_reg": losses["loss_split_reg"] * self.smooth_l1_loss_weight})
        if "loss_split_cls" in losses.keys():
            losses.update({"loss_split_cls": losses["loss_split_cls"] * self.cross_entropy_loss_weight})
        if "loss_split_giou" in losses.keys():
            losses.update({"loss_split_giou": losses["loss_split_giou"] * self.giou_weight})
        return losses


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        if inspect.ismethod(cls._init_maskiou_head):
            ret.update(cls._init_maskiou_head(cfg))

        ret["use_giou"] = cfg.MODEL.ROI_BOX_HEAD.USE_GIOU
        ret["loss_weights"] = cfg.MODEL.ROI_HEADS.LOSS_WEIGHTS
        ret['maskiou_on'] = cfg.MODEL.MASKIOU_ON

        ret["refine_cls"] = cfg.MODEL.ROI_HEADS.REFINE_CLS
        ret['refine_cls_type'] = cfg.MODEL.ROI_HEADS.REFINE_METHOD
        ret['dml_embedding_size'] = cfg.MODEL.ROI_HEADS.DML_EMBEDDING_SIZE
        ret['wha_head_on'] = cfg.MODEL.ROI_HEADS.WHA_HEAD
        ret['cls_head_bn_blocks'] = cfg.MODEL.ROI_HEADS.CONV_BN_BLOCKS

        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        roi_box_head_type = cfg.MODEL.ROI_HEADS.BOX_HEAD_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        if roi_box_head_type == 'shared':
            # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
            # They are used together so the "box predictor" layers should be part of the "box head".
            # New subclasses of ROIHeads do not need "box predictor"s.
            box_head = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
            return {
                "box_in_features": in_features,
                "box_pooler": box_pooler,
                "box_head": box_head,
                "box_predictor": box_predictor,
                "roi_box_head_type": roi_box_head_type,
            }
        elif roi_box_head_type == 'double':
            split_cls_head = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
            split_reg_head = ConvRegBoxHead()
            box_predictor = SplittedFastRCNNOutputLayers(cfg, split_cls_head.output_shape)
            return {
                "box_in_features": in_features,
                "box_pooler": box_pooler,
                "split_cls_head": split_cls_head,
                "split_reg_head": split_reg_head,
                "box_predictor": box_predictor,
                "roi_box_head_type": roi_box_head_type,
                "box_head": None, # for compatibility with ROIHeads constructor
            }
        elif roi_box_head_type == 'both':
            box_head = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)

            split_cls_head = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
            split_reg_head = ConvRegBoxHead()
            split_box_predictor = SplittedFastRCNNOutputLayers(cfg, split_cls_head.output_shape)

            return {
                "box_in_features": in_features,
                "box_pooler": box_pooler,

                "split_cls_head": split_cls_head,
                "split_reg_head": split_reg_head,

                "box_head": box_head,
                "box_predictor": (box_predictor, split_box_predictor),

                "roi_box_head_type": roi_box_head_type,
            }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        mask_attention    = cfg.MODEL.ROI_MASK_HEAD.ATTENTION
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {
            "mask_in_features": in_features,
            "mask_pooler": ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
                ),
            "mask_head": build_mask_head(
                cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)),
            'mask_attention': mask_attention}

        return ret

    @classmethod
    def _init_maskiou_head(cls, cfg):
        if not cfg.MODEL.MASKIOU_ON:
            return {"maskiou_on": False}
        ret = {
            "maskiou_on": True,
            'maskiou_head': build_maskiou_head(cfg),
            'maskiou_weight': cfg.MODEL.MASKIOU_LOSS_WEIGHT
        }

        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {
            "keypoint_in_features": in_features,
            "keypoint_pooler": ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
                ),
            "keypoint_head": build_keypoint_head(
                cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))
        }
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        if self.training:
            losses, pooled_box_features = self._forward_box(features, proposals)

            if self.wha_head_on:
                losses.update(self._forward_wha_regressor(pooled_box_features, proposals))

            if self.refine_cls:
                if self.refine_cls_type == 'dml':
                    losses.update(self._forward_metric(features, proposals))
                if self.refine_cls_type == 'conv':
                    losses.update(self._forward_refine_score_box(pooled_box_features, instances=proposals))

            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            if self.mask_on:
                if self.maskiou_on:
                    if self.mask_attention:
                        forward_res, mask_features, fg_idxs = self._forward_mask(features, proposals)
                        (mask_loss, selected_mask, labels, maskiou_targets, out_mask) = forward_res
                        fg_idxs = torch.cat(fg_idxs)
                        split_head_losses = self._forward_attention(out_mask, pooled_box_features, fg_idxs, proposals)
                        losses.update(split_head_losses)
                    else:
                        forward_res, mask_features = self._forward_mask(features, proposals)
                        (mask_loss, selected_mask, labels, maskiou_targets) = forward_res
                    maskiou_loss = self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets)
                    losses.update(mask_loss)
                    losses.update(maskiou_loss)
                else:
                    if self.mask_attention:
                        (mask_loss, out_mask), fg_idxs = self._forward_mask(features, proposals)
                        fg_idxs = torch.cat(fg_idxs)
                        split_head_losses = self._forward_attention(out_mask, pooled_box_features, fg_idxs, proposals)
                        losses.update(mask_loss)
                        losses.update(split_head_losses)
                    else:
                        losses.update(self._forward_mask(features, proposals))

            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, pooled_box_features, filter_indxs = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.

            # if self.wha_head_on:
            #     (self._forward_wha_regressor(pooled_box_features, proposals))

            if self.refine_cls:
                if self.refine_cls_type == 'dml':
                    pred_instances = self._forward_metric(features, pred_instances)
                if self.refine_cls_type == 'conv':
                    pred_instances = self._forward_refine_score_box(pooled_box_features[filter_indxs], instances=pred_instances)

            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances], attention_features=None
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.maskiou_on:
            instances, mask_features = self._forward_mask(features, instances)
            instances = self._forward_maskiou(mask_features, instances)
        else:
            instances = self._forward_mask(features, instances)

        instances = self._forward_keypoint(features, instances)

        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        features = [features[f] for f in self.box_in_features]
        pooled_box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        if self.pooler_attention_on:
            pooled_box_features = self.pooler_box_attention(pooled_box_features)

        if self.roi_box_head_type == "both":
            shared_box_features = self.box_head(pooled_box_features)
            shared_predictions = self.box_predictor(shared_box_features)
            del shared_box_features
            box_reg_features = self.split_reg_head(pooled_box_features)
            box_cls_features = self.split_cls_head(pooled_box_features)
            double_predictions = self.box_predictor(box_reg_features, box_cls_features)
            del box_reg_features
            del box_cls_features

            if self.training:
                losses = self.box_predictor[0].losses(shared_predictions, proposals)
                losses.update(self.box_predictor[1].losses(double_predictions, proposals))
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        shared_pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            shared_predictions, proposals
                        )
                        double_pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            double_predictions, proposals
                        )
                        pred_boxes = torch.cat([shared_pred_boxes, double_pred_boxes])

                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                losses = self.reweight_losses(losses)

                return losses, pooled_box_features
            else:
                pred_instances, _ = self.box_predictor[0].inference(shared_predictions, proposals)
                double_pred_instances, _ = self.box_predictor[1].inference(double_predictions, proposals)
                merged_pred_intances = []
                for shared_inst, double_inst in zip(pred_instances, double_pred_instances):
                    merged_pred_intances.append(Instances.cat([shared_inst, double_inst]))

                return merged_pred_intances, pooled_box_features

        else:
            if self.roi_box_head_type == 'shared':
                box_features = self.box_head(pooled_box_features)
                predictions = self.box_predictor(box_features)
                del box_features
            else: # double
                box_reg_features = self.split_reg_head(pooled_box_features)
                box_cls_features = self.split_cls_head(pooled_box_features)
                predictions = self.box_predictor(box_reg_features, box_cls_features)
                del box_reg_features
                del box_cls_features
            if self.training:
                losses = self.box_predictor.losses(predictions, proposals)
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                losses = self.reweight_losses(losses)

                return losses, pooled_box_features
            else:
                pred_instances, filter_indx = self.box_predictor.inference(predictions, proposals)
                return pred_instances, pooled_box_features, filter_indx

    def _forward_wha_regressor(self, features, instances):
        if not self.wha_head_on:
            return {} if self.training else instances

        if self.training:
            with torch.no_grad():
                image_sizes = [p.image_size for p in instances]
                gt_boxes = [p.gt_boxes for p in instances]
                gt_areas = [p.gt_boxes.area() for p in instances]
                norm_gt_centers = []
                norm_gt_areas = []
                for im_size, im_gt_boxes, im_gt_area in zip(image_sizes, gt_boxes, gt_areas):
                    im_gt_tensor = im_gt_boxes.tensor
                    norm_gt_centers.append(torch.stack(
                        [im_gt_tensor[:, 0] / im_size[1], im_gt_tensor[:, 1] / im_size[0], im_gt_tensor[:, 2] / im_size[1],
                         im_gt_tensor[:, 3] / im_size[0]], dim=1))
                    norm_gt_areas.append(im_gt_area / (im_size[0] * im_size[1]))

                norm_gt_centers = torch.cat(norm_gt_centers)
                norm_gt_areas = torch.cat(norm_gt_areas)
                norm_gts = torch.stack([(norm_gt_centers[:, 2] - norm_gt_centers[:, 0]) / 2,
                                        (norm_gt_centers[:, 2] - norm_gt_centers[:, 0]) / 2, norm_gt_areas], dim=1)

            wha_loss = self.wha_head(features, norm_gts)
            return wha_loss
        else:
            return instances

    def _forward_refine_score_box(self, pooled_box_features, instances=None):

        if not self.refine_cls:
            return {} if self.training else instances

        if self.training:
            targets = torch.cat([proposal.get('gt_classes') for proposal in instances])
            return self.refine_head(pooled_box_features, targets)
        else:
            return self.refine_head(pooled_box_features, instances)

    def _forward_metric(self, features, instances):
        if not self.metric_head:
            return {} if self.training else instances

        features = [features[f] for f in self.box_in_features]
        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            mask_features = self.box_pooler(features, proposal_boxes)

            targets = torch.cat([proposal.get('gt_classes') for proposal in instances])

            mask = self.get_metric_samples(targets)

            embeddings = self.refine_head(mask_features[mask], targets[mask])
            self.proto_centroids = recompute_proto_centroids(self.proto_centroids, embeddings, targets[mask])
            torch.save(self.proto_centroids, "centroids.pkl")

            miner_output = self.refine_miner(embeddings, targets[mask])
            loss_dict = self.metric_loss(embeddings, targets[mask], miner_output)
            return {"loss_metric_cls": loss_dict / embeddings.size(0)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.box_pooler(features, pred_boxes)
            pred_embeddings, pred_instances = self.refine_head(mask_features, instances)

            if pred_embeddings is not None:
                dist_to_centroids = torch.cdist(pred_embeddings, self.proto_centroids)
                probs_to_cls = dist_to_centroids.sigmoid()

                n_pred_instances = []
                for idx_instance, instance in enumerate(pred_instances):
                    new_scores = []
                    for idx_box, (pred_cls, score) in enumerate(zip(instance.pred_classes, instance.scores)):
                        dml_pred_class = probs_to_cls[idx_instance * len(instance) + idx_box].max(dim=0)[1] + 1
                        if pred_cls != dml_pred_class:
                            new_scores.append(
                                score * probs_to_cls[idx_instance * len(instance) + idx_box, pred_cls - 1])
                        else:
                            new_scores.append(score)
                    instance.scores = torch.tensor(new_scores, dtype=instance.scores.dtype,
                                                   device=instance.scores.device)
                    n_pred_instances.append(instance)
                pred_instances = n_pred_instances

            return pred_instances

    def get_metric_samples(self, targets):
        """
        Function that returns a boolean mask with the selected elements
        :param targets:
        """
        with torch.no_grad():
            out_targets = torch.zeros_like(targets).bool()
            classes_on_targets, num_per_class = targets.unique(return_counts=True)

            min_elements = num_per_class.min()
            for cls in classes_on_targets:
                out_targets[(targets == cls).nonzero().view(-1, 1)[:min_elements]] = True
            return out_targets

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.mask_in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            if self.mask_attention:
                proposals, fg_idxs = select_foreground_proposals(instances, self.num_classes)
            else:
                proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            if self.maskiou_on:
                if self.mask_attention:
                    return self.mask_head(mask_features, proposals), mask_features, fg_idxs
                return self.mask_head(mask_features, proposals), mask_features
            elif self.mask_attention:
                return self.mask_head(mask_features, proposals), fg_idxs
            else:
                return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            self.mask_head(mask_features, instances)

            if self.maskiou_on:
                return instances, mask_features
            else:
                return instances

    def _forward_maskiou(self, features, instances, selected_mask=None, labels=None, targets=None):
        """
        Forward logic of the mask iou prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, calibrate instances' scores.
        """
        if not self.maskiou_on:
            return {} if self.training else instances

        if self.training:
            pred_maskiou = self.maskiou_head(features, selected_mask)
            return {"loss_maskiou": mask_iou_loss(labels, pred_maskiou, targets, self.maskiou_weight)}

        else:
            if features.size(0) > 0:
                pred_maskiou = self.maskiou_head(features, torch.cat([i.pred_masks for i in instances], 0))
                mask_iou_inference(instances, pred_maskiou)
            return instances

    def _forward_attention(self, predicted_masks, pooled_box_features, fg_idxs, proposals, features=None):
        ret_features = False
        if self.training:
            if False: # attention layers
                filters = torch.zeros_like(pooled_box_features, dtype=pooled_box_features.dtype, device=pooled_box_features.device)
                filters[fg_idxs] = self.attention_layers(predicted_masks, pooled_box_features[fg_idxs])
                box_reg_features = pooled_box_features + filters
                box_cls_features = pooled_box_features

            elif False: # angostic mask attention
                box_reg_features = pooled_box_features
                box_cls_features = pooled_box_features
                cls_mod_filters = torch.ones_like(pooled_box_features, dtype=pooled_box_features.dtype, device=pooled_box_features.device)
                down_masks = torch.nn.functional.max_pool2d(predicted_masks, kernel_size=4)
                cls_mod_filters[fg_idxs] = down_masks

                box_cls_features = pooled_box_features * cls_mod_filters
            else:
                box_reg_features = pooled_box_features
                box_cls_features = pooled_box_features

            box_reg_features = self.split_reg_head(box_reg_features)
            box_cls_features = self.split_cls_head(box_cls_features)
            # box_cls_features = box_reg_features
            predictions = self.box_predictor(box_reg_features, box_cls_features)
            del box_reg_features
            del box_cls_features

            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.reweight_losses(losses)

            if ret_features:
                return losses, pooled_box_features
            return losses
        else:
            pooled_box_features = [pooled_box_features[f] for f in self.box_in_features]
            pooled_box_features = self.box_pooler(pooled_box_features, [x.proposal_boxes for x in proposals])

            box_reg_features = self.split_reg_head(pooled_box_features)
            box_cls_features = self.split_cls_head(pooled_box_features)

            predictions = self.box_predictor(box_reg_features, box_cls_features)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            return pred_instances, pooled_box_features

    def _forward_keypoint(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.keypoint_in_features]

        if self.training:
            # The loss is defined on positive proposals with >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)
