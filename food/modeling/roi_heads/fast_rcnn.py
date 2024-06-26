# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Union
import scipy as sp

import numpy as np
import torch
import torch.distributions as dists
from detectron2.config import configurable
from detectron2.layers import (ShapeSpec, batched_nms, cat,
                               nonzero_tuple)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

# from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
#                                                      _log_classification_stats)

from detectron2.structures import Boxes, Instances, pairwise_iou
# from detectron2.structures.boxes import matched_boxlist_iou
#  fast_rcnn_inference)
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import giou_loss, smooth_l1_loss
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import functional as F
from ...layers.soft_nms import batched_soft_nms
from ...layers import MLP
from ...losses import ICLoss, UpLoss


ROI_BOX_OUTPUT_LAYERS_REGISTRY = Registry("ROI_BOX_OUTPUT_LAYERS")
ROI_BOX_OUTPUT_LAYERS_REGISTRY.__doc__ = """
ROI_BOX_OUTPUT_LAYERS
"""

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    logits: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float = 1.0,
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, logits, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    logits,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
):
    valid_mask = torch.isfinite(boxes).all(
        dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    scores = scores[:, :-1]
    second_scores = scores
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        second_scores = second_scores[filter_inds[:, 0], :]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    logits = logits[filter_inds[:, 0], :]
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, logits, filter_inds = boxes[keep], scores[keep], logits[keep], filter_inds[keep]

    # u = 22/torch.sum(torch.exp(logits)+1, dim=1)
    # keep = u<0.015
    # keep1 = filter_inds[:, 1]==20
    # keep2 = keep
    # for i in range(len(keep)):
    #     if keep[i] & keep1[i]:
    #         keep2[i] = True
    #     else:
    #         keep2[i] = False
    # keep = ~keep2
    # boxes, scores, logits, filter_inds = boxes[keep], scores[keep], logits[keep], filter_inds[keep]

    # # uncertain label reassignment.
    # second_pred_classes = filter_inds[:, 1]
    # for un_id in range(len(second_pred_classes)):
    #     if second_pred_classes[un_id] == 80:
    #         un_score = second_scores[un_id]
    #         no_un_score = un_score[:-1]
    #         un_label = no_un_score.argmax(dim=-1)
    #         filter_inds[un_id,1] = un_label

    # apply nms between known classes and unknown class for visualization.
    # vis_iou_thr = 0.1
    uncertain_id = 80
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(
            boxes, scores, filter_inds, uncertain_id, iou_thr=vis_iou_thr)
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

def unknown_aware_nms(boxes, scores, labels, ukn_class_id=20, iou_thr=0.9):
    u_inds = labels[:, 1] == ukn_class_id
    k_inds = ~u_inds
    if k_inds.sum() == 0 or u_inds.sum() == 0:
        return boxes, scores, labels

    k_boxes, k_scores, k_labels = boxes[k_inds], scores[k_inds], labels[k_inds]
    u_boxes, u_scores, u_labels = boxes[u_inds], scores[u_inds], labels[u_inds]

    ious = pairwise_iou(Boxes(k_boxes), Boxes(u_boxes))
    mask = torch.ones((ious.size(0), ious.size(1), 2), device=ious.device)
    inds = (ious > iou_thr).nonzero()
    if not inds.numel():
        return boxes, scores, labels

    for [ind_x, ind_y] in inds:
        if k_scores[ind_x] >= u_scores[ind_y]:
            mask[ind_x, ind_y, 1] = 0
        else:
            mask[ind_x, ind_y, 0] = 0

    k_inds = mask[..., 0].mean(dim=1) == 1
    u_inds = mask[..., 1].mean(dim=0) == 1

    k_boxes, k_scores, k_labels = k_boxes[k_inds], k_scores[k_inds], k_labels[k_inds]
    u_boxes, u_scores, u_labels = u_boxes[u_inds], u_scores[u_inds], u_labels[u_inds]

    boxes = torch.cat([k_boxes, u_boxes])
    scores = torch.cat([k_scores, u_scores])
    labels = torch.cat([k_labels, u_labels])

    return boxes, scores, labels


logger = logging.getLogger(__name__)


def build_roi_box_output_layers(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS
    return ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(name)(cfg, input_shape)

class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction="mean"
        )

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1)
            .expand(num_pred, K, B)
            .reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        # a = self.pred_class_logits[:, :-2]
        # b = self.pred_class_logits[:, -1:]
        # self.pred_class_logits = torch.cat([a, b], dim=1)
        probs = F.softmax(self.pred_class_logits, dim=-1)

        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        logits = self.pred_class_logits
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            logits,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )

@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class CosineFastRCNNOutputLayers(FastRCNNOutputLayers):

    @configurable
    def __init__(
        self,
        *args,
        scale: int = 20,
        vis_iou_thr: float = 1.0,
        number_classes: int = 20,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.num_classes = number_classes
        self.cls_score = nn.Linear(
            self.cls_score.in_features, self.num_classes + 1, bias=False)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        # scaling factor
        self.scale = scale
        self.vis_iou_thr = vis_iou_thr


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['scale'] = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        ret['vis_iou_thr'] = cfg.MODEL.ROI_HEADS.VIS_IOU_THRESH
        ret['number_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def forward(self, feats):

        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):

        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.vis_iou_thr,
        )

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        if not len(proposals):
            return []
        proposal_deltas = predictions[1]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat(
            [p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class PROSERFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    """PROSER
    """
    @configurable
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.proser_weight = 0.1

    def get_proser_loss(self, scores, gt_classes):
        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != gt_classes[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes-1)
        mask_scores = torch.gather(scores, 1, mask)

        targets = torch.zeros_like(gt_classes)
        fg_inds = gt_classes != self.num_classes
        targets[fg_inds] = self.num_classes-2
        targets[~fg_inds] = self.num_classes-1

        loss_cls_proser = F.cross_entropy(mask_scores, targets)
        return {"loss_cls_proser": self.proser_weight * loss_cls_proser}

    def losses(self, predictions, proposals, input_features=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(
                proposals) else torch.empty(0)
        )
        # _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes")
                  else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls_ce": F.cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        losses.update(self.get_proser_loss(scores, gt_classes))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class DropoutFastRCNNOutputLayers(CosineFastRCNNOutputLayers):

    @configurable
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.dropout = nn.Dropout(p=0.5)
        self.entropy_thr = 0.25

    def forward(self, feats, testing=False):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        if testing:
            self.dropout.train()
            x_normalized = self.dropout(x_normalized)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas

    def inference(self, predictions: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions[0], proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_probs(
        self, predictions: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        # mean of multiple observations
        scores = torch.stack([pred[0] for pred in predictions], dim=-1)
        scores = scores.mean(dim=-1)
        # threshlod by entropy
        norm_entropy = dists.Categorical(scores.softmax(
            dim=1)).entropy() / np.log(self.num_classes)
        inds = norm_entropy > self.entropy_thr
        max_scores = scores.max(dim=1)[0]
        # set those with high entropy unknown objects
        scores[inds, :] = 0.0
        scores[inds, self.num_classes-1] = max_scores[inds]

        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)

        if self._do_cls_dropout:
            x = F.dropout(x, self._dropout_ratio, training=self.training)
        scores = self.cls_score(x)

        return scores, proposal_deltas


class ClipFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            is_base_train,
            num_base_classes,
            num_known_classes,
            max_iter: int,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            soft_nms_enabled=False,
            soft_nms_method="gaussian",
            soft_nms_sigma=0.5,
            soft_nms_prune=0.001,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            clip_cls_emb: tuple = (False, None),
            no_box_delta: bool = False,
            bg_align_loss_weight: None,
            multiply_rpn_score: tuple = (False, False),
            openset_test: None,
            topk,
            sampling_ratio,
            topm,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.soft_nms_enabled = soft_nms_enabled
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_prune = soft_nms_prune
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        # RegionCLIP
        self.num_classes = num_classes
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        self.use_clip_cls_emb = clip_cls_emb[0]
        if self.use_clip_cls_emb:  # use CLIP text embeddings as classifier's weights
            input_size = clip_cls_emb[3] if clip_cls_emb[2] in ['CLIPRes5ROIHeads',
                                                                'CLIPStandardROIHeads'] else input_size
            # text_emb_require_grad = False
            self.use_bias = False
            self.temperature = openset_test[2]  # 0.01 is default for CLIP

            # class embedding
            self.un_score = nn.Linear(input_size, 1, bias=self.use_bias)
            nn.init.normal_(self.un_score.weight, std=0.01)
            self.un_score.weight.requires_grad = True

            # background embedding
            self.text_bg_score = nn.Linear(input_size, 1, bias=self.use_bias)
            nn.init.normal_(self.text_bg_score.weight, std=0.01)
            self.text_bg_score.weight.requires_grad = True


        else:  # regular classification layer
            self.cls_score = nn.Linear(input_size, num_classes + 1)  # one background class (hence + 1)
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)

        # box regression layer
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        # training options
        self.align_loss_weight = None
        if bg_align_loss_weight is not None:  # loss weigh for bg class
            self.align_loss_weight = torch.ones(num_classes + 1)

            self.align_loss_weight[-1] = bg_align_loss_weight
        self.focal_scaled_loss = openset_test[3]  # focal scaling
        # inference options
        self.no_box_delta = no_box_delta  # box delta after regression
        self.multiply_rpn_score = multiply_rpn_score[0]
        self.vis = multiply_rpn_score[1]  # if enabled, visualize scores before multiplying RPN scores


        self.max_iters = max_iter

        self.num_known_classes = num_known_classes
        self.num_base_classes = num_base_classes


        self.topk = topk
        self.sampling_ratio = sampling_ratio
        self.topm = topm
        self.Up_loss = UpLoss(
            self.num_classes,
            self.num_base_classes,
            sampling_metric="gradient",
            sampling_ratio=self.sampling_ratio,
            topk=self.topk,
            alpha=1.0,
        )

        self.ic_loss_loss = ICLoss(tau=0.1)
        self.ic_loss_out_dim = 128
        self.ic_loss_queue_size = 256
        self.ic_loss_in_queue_size = 16
        self.encoder = MLP(self.un_score.in_features, self.ic_loss_out_dim)
        self.register_buffer('queue', torch.zeros(
            self.num_known_classes, self.ic_loss_queue_size, self.ic_loss_out_dim))
        self.register_buffer('queue_label', torch.empty(
            self.num_known_classes, self.ic_loss_queue_size).fill_(-1).long())
        self.register_buffer('queue_ptr', torch.zeros(
            self.num_known_classes, dtype=torch.long))
        self.register_buffer('queue_mean', torch.zeros(
            self.num_known_classes, self.ic_loss_out_dim, dtype=torch.long))
        self.ic_loss_weight = 0.1


        ####
        self.is_base_train = is_base_train
        self.K = 1


    @classmethod
    def from_config(cls, cfg, input_shape):
        # if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN":
        #     assert cfg.MODEL.CLIP.NO_BOX_DELTA is False
        return {
            "is_base_train": cfg.MODEL.CLIP.BASE_TRAIN,
            "num_base_classes": cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES,
            "num_known_classes": cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "soft_nms_enabled": cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED,
            "soft_nms_method": cfg.MODEL.ROI_HEADS.SOFT_NMS_METHOD,
            "soft_nms_sigma": cfg.MODEL.ROI_HEADS.SOFT_NMS_SIGMA,
            "soft_nms_prune": cfg.MODEL.ROI_HEADS.SOFT_NMS_PRUNE,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # RegionCLIP
            "clip_cls_emb": (
            cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER, cfg.MODEL.CLIP.TEXT_EMB_PATH, cfg.MODEL.ROI_HEADS.NAME,
            cfg.MODEL.CLIP.TEXT_EMB_DIM),
            "no_box_delta": cfg.MODEL.CLIP.NO_BOX_DELTA or cfg.MODEL.CLIP.CROP_REGION_TYPE == 'GT',
            "bg_align_loss_weight": cfg.MODEL.CLIP.BG_CLS_LOSS_WEIGHT,
            "multiply_rpn_score": (cfg.MODEL.CLIP.MULTIPLY_RPN_SCORE, cfg.MODEL.CLIP.VIS),
            "openset_test": (cfg.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES, cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH, \
                             cfg.MODEL.CLIP.CLSS_TEMP, cfg.MODEL.CLIP.FOCAL_SCALED_LOSS),
            "topk": cfg.UPLOSS.TOPK,
            "sampling_ratio": cfg.UPLOSS.SAMPLING_RATIO,
            "topm": cfg.UPLOSS.TOPM,
            # fmt: on
        }


    def forward(self, x, text_emb):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)


        # use clip text embeddings as classifier's weights
        if self.use_clip_cls_emb:
            normalized_x = F.normalize(x, p=2.0, dim=1)

            align_scores = normalized_x @ F.normalize(text_emb, p=2.0, dim=1).t()



            ###
            if self.use_bias:
                align_scores += self.text_score.bias



            un_scores = self.un_score(normalized_x)

            # background class (zero embeddings)
            align_bg_score = self.text_bg_score(normalized_x)

            if self.use_bias:
                align_bg_score += self.text_bg_score.bias

            align_scores = torch.cat((align_scores, un_scores, align_bg_score), dim=1) #512*(num_cls*K+2)


            scores = align_scores / self.temperature


            ####
        # regular classifier
        else:
            scores = self.cls_score(x)

        ###
        mlp_feat = self.encoder(x)

        # mlp_feat = []
        ###

        # box regression
        proposal_deltas = self.bbox_pred(x)
        # local_scores = None
        return scores, proposal_deltas, mlp_feat

    def local_feat_forward(self, x_local, text_emb):
        local_norm = x_local / x_local.norm(dim=-1, keepdim=True)
        local_scores = (local_norm @ F.normalize(text_emb, p=2.0, dim=1).t())
        un_local = self.un_score(local_norm)
        local_bg_score = self.text_bg_score(local_norm)
        local_scores = torch.cat((local_scores, un_local, local_bg_score), dim=2)
        local_scores = local_scores / self.temperature

        return local_scores


    def losses(self, predictions, proposals, squarescores, cat_feat):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, mlp_feat = predictions



        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)


        # loss weights
        if self.align_loss_weight is not None and self.align_loss_weight.device != scores.device:
            self.align_loss_weight = self.align_loss_weight.to(scores.device)
        if self.focal_scaled_loss is not None:


            loss_align = self.focal_loss(scores, gt_classes, gamma=self.focal_scaled_loss)

        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean") if self.align_loss_weight is None else \
                cross_entropy(scores, gt_classes, reduction="mean", weight=self.align_loss_weight)
        losses = {

            "loss_align": loss_align,

            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        objectness_logits = [x.objectness_logits for x in proposals]
        # objectness = torch.sigmoid(torch.cat(objectness_logits))
        objectness = (torch.cat(objectness_logits))
        mask = objectness>1
        objectness[mask]=1.0
        ious = cat([p.iou for p in proposals], dim=0)




        if not self.is_base_train:
            losses.update(self.get_up_loss(scores, gt_classes, squarescores, objectness, ious))

        if not self.is_base_train:
            losses.update(self.get_feature_loss(cat_feat))

        ####

        self._dequeue_and_enqueue(
            mlp_feat, gt_classes, ious)
        losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))
        ####



        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def get_feature_loss(self, cat_feat):
        cat_local_scores, cat_labels, cat_ious, cat_grad_map, cat_scores = cat_feat
        fg_inds = cat_labels != self.num_classes
        fg_local_scores = cat_local_scores[fg_inds]
        bg_local_scores = cat_local_scores[~fg_inds]
        fg_grad_map = cat_grad_map[fg_inds]
        bg_grad_map = cat_grad_map[~fg_inds]

        fg_labels = cat_labels[fg_inds]


        ab_num = self.topm
        # print("m", ab_num)




        select_fg_pos = fg_local_scores[torch.arange(fg_grad_map.shape[0]).unsqueeze(1),
                    torch.topk(fg_grad_map[:, :-1], k=ab_num, dim=1, largest=True)[1], :]
        select_bg_neg = bg_local_scores[torch.arange(bg_grad_map.shape[0]).unsqueeze(1),
                                torch.topk(bg_grad_map[:, :-1], k=ab_num, dim=1, largest=True)[1], :]

        probs_fg = select_fg_pos[:, :, :self.num_known_classes].softmax(dim=-1)
        probs_bg = select_bg_neg[:, :, :self.num_known_classes].softmax(dim=-1)
        entropy_fg = -torch.sum(probs_fg * torch.log2(probs_fg), dim=-1) / torch.log2(torch.tensor(self.num_known_classes))
        entropy_bg = -torch.sum(probs_bg * torch.log2(probs_bg), dim=-1) / torch.log2(torch.tensor(self.num_known_classes))



        fg_num = select_fg_pos.size(0)
        bg_num = select_bg_neg.size(0)

        targets_fg = torch.zeros(fg_num, ab_num, self.num_classes, device=select_fg_pos.device)
        targets_bg = torch.zeros(bg_num, ab_num, self.num_classes, device=select_bg_neg.device)
        # print(entropy_fg)
        # print(entropy_bg)

        for i in range(fg_num):
            for j in range(ab_num):
                targets_fg[i, j, fg_labels[i]] = 1
                targets_fg[i, j, -1] = entropy_fg[i,j]
        for i in range(bg_num):
            for j in range(ab_num):
                targets_bg[i, :, -1] = entropy_bg[i,j]

        loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
        loss_fg = loss_func(select_fg_pos[:, :, :-1], targets_fg)
        loss_bg = loss_func(select_bg_neg[:, :, :-1], targets_bg)


        loss_feature = (loss_fg + loss_bg) / 2.0




        storage = get_event_storage()
        lamda = 0.0001
        decay_weight = torch.exp(torch.log(torch.tensor(lamda)) * (1 - (storage.iter / self.max_iters)))

        return {"loss_feature": decay_weight * loss_feature}

    def get_up_loss(self, scores, gt_classes, squarescores, objectness,ious):
        # start up loss after several warmup iters
        storage = get_event_storage()
        lamda = 0.0001

        decay_weight = torch.exp(torch.log(torch.tensor(lamda)) * (1-(storage.iter / self.max_iters)))
        # decay_weight = (torch.exp(torch.log(torch.tensor(lamda)) * (1 - (storage.iter / self.max_iters))) - lamda)*(1/(1-lamda))

        if storage.iter >= 0:
            loss_cls_up = self.Up_loss(scores, gt_classes, squarescores, objectness, ious)
        else:
            loss_cls_up = scores.new_tensor(0.0)

        return {"loss_up": decay_weight * loss_cls_up}

    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > 0.5) & (
            gt_classes != self.num_classes)
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return {"loss_cls_ic": self.ic_loss_weight * decay_weight * loss_ic_loss}

    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 1. gather variable
        # feat = self.concat_all_gather(feat)
        # gt_classes = self.concat_all_gather(gt_classes)
        # ious = self.concat_all_gather(ious)
        # 2. filter by iou and obj, remove bg
        keep = (ious > iou_thr) & (gt_classes != self.num_classes)
        feat, gt_classes = feat[keep], gt_classes[keep]

        for i in range(self.num_known_classes):
            ptr = int(self.queue_ptr[i])
            cls_ind = gt_classes == i
            cls_feat, cls_gt_classes = feat[cls_ind], gt_classes[cls_ind]
            # 3. sort by similarity, low sim ranks first
            cls_queue = self.queue[i, self.queue_label[i] != -1]
            _, sim_inds = F.cosine_similarity(
                cls_feat[:, None], cls_queue[None, :], dim=-1).mean(dim=1).sort()
            top_sim_inds = sim_inds[:self.ic_loss_in_queue_size]
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]
            # 4. in queue
            batch_size = cls_feat.size(
                0) if ptr + cls_feat.size(0) <= self.ic_loss_queue_size else self.ic_loss_queue_size - ptr
            self.queue[i, ptr:ptr+batch_size] = cls_feat[:batch_size].data
            self.queue_label[i, ptr:ptr + batch_size] = cls_gt_classes[:batch_size].data

            ptr = ptr + batch_size if ptr + batch_size < self.ic_loss_queue_size else 0
            self.queue_ptr[i] = ptr


    def concat_all_gather(self, tensor):
        world_size = comm.get_world_size()
        # single GPU, directly return the tensor
        if world_size == 1:
            return tensor
        # multiple GPUs, gather tensors
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def focal_loss(self, inputs, targets, gamma=0.5, reduction="mean"):
        """Inspired by RetinaNet implementation"""
        if targets.numel() == 0 and reduction == "mean":
            return input.sum() * 0.0  # connect the gradient


        # focal scaling
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = F.softmax(inputs, dim=-1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        loss = ce_loss * ((1 - p_t) ** gamma)
        # loss = ce_loss

        # bg loss weight
        if self.align_loss_weight is not None:
            loss_weight = torch.ones(loss.size(0)).to(p.device)
            loss_weight[targets == self.num_classes] = self.align_loss_weight[-1].item()
            loss = loss * loss_weight

        if reduction == "mean":
            loss = loss.mean()

        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        # optional: multiply class scores with RPN scores
        scores_bf_multiply = scores  # as a backup for visualization purpose
        if self.multiply_rpn_score and not self.training:
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]
        return fast_rcnn_inference_clip(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.soft_nms_enabled,
            self.soft_nms_method,
            self.soft_nms_sigma,
            self.soft_nms_prune,
            self.test_topk_per_image,
            scores_bf_multiply=scores_bf_multiply,
            vis=True if self.vis else False,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas,t = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        # don't apply box delta, such as GT boxes
        if self.no_box_delta:
            predict_boxes = proposal_boxes
        # apply box delta
        else:
            predict_boxes = self.box2box_transform.apply_deltas(
                proposal_deltas,
                proposal_boxes,
            )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ , t= predictions

        # max_scores,_ = torch.max(scores,dim=1)
        # alpha = torch.sum(max_scores) / torch.sum(torch.abs(scores[:, -2]))
        # scores[:, -2] = torch.abs(scores[:, -2]) * alpha
        # scores[:,-2] = torch.sigmoid(scores[:,-2])
        # mask = scores[:,-2]>0.5
        # scores[:, -2][mask] = 100

        # mask = self.queue_label != -1
        # prototype = torch.zeros((self.num_classes-1,self.ic_loss_out_dim), device="cuda")
        # for i in range(self.num_classes-1):
        #     prototype[i]=self.queue[i][mask[i]].mean(dim=0)

        # prototype_score = t @ prototype.T
        num_inst_per_image = [len(p) for p in proposals]

        # cal_scores = scores[:, :-2].clone().detach()
        # split_scores = torch.stack(cal_scores.split(self.num_known_classes, dim=1), dim=0)
        # # split_scores = cal_scores.reshape(self.K, cal_scores.shape[0],
        # #                                   self.num_base_classes if self.is_base_train
        # #                                   else self.num_known_classes)
        # mask = split_scores<0
        # # print(torch.sum(mask))
        # split_scores[mask]=0.00001
        # # split_scores = split_scores + 100
        #
        # sum_split_scores = torch.sum(split_scores, dim=0)
        # norm_split_scores = split_scores / sum_split_scores
        # scores_weight = torch.cat(torch.split(norm_split_scores, 1, dim=0), dim=2).squeeze(0)
        # # scores_weight = norm_split_scores.reshape(cal_scores.shape[0],
        # #                                           self.num_base_classes * self.K if self.is_base_train
        # #                                           else self.num_known_classes * self.K)
        # scores_weight = torch.concat(
        #     (scores_weight, torch.ones(cal_scores.shape[0], 2, device=scores.device)), dim=1)
        #
        # exp_scores = torch.exp(scores)
        #
        # final_scores = exp_scores * scores_weight.data
        #
        # final_cal_scores = final_scores[:, :-2].clone().detach()
        #
        # final_split_scores = torch.stack(final_cal_scores.split(self.num_known_classes, dim=1), dim=0)
        # # if self.is_base_train:
        # #     final_split_scores[:, :, self.num_base_classes:] = 0
        # final_sum_scores = torch.sum(final_split_scores, dim=0)
        # inference_scores = torch.concat(
        #     (final_sum_scores, final_scores[:, -2:]), dim=1)
        # a = torch.sum(inference_scores, dim=1).view(-1, 1)
        # probs = inference_scores / a
        probs = F.softmax(scores, dim=-1)

        return probs.split(num_inst_per_image, dim=0)

def fast_rcnn_inference_clip(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
    scores_bf_multiply: List[torch.Tensor],
    vis=False,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        soft_nms_enabled (bool): Indicate to use soft non-maximum suppression.
        soft_nms_method: (str): One of ['gaussian', 'linear', 'hard']
        soft_nms_sigma: (float): Sigma for gaussian soft nms. Value in (0, inf)
        soft_nms_prune: (float): Threshold for pruning during soft nms. Value in [0, 1]
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image_clip(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh,
            soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune, topk_per_image, s_bf_per_img, vis
        )
        for scores_per_image, boxes_per_image, image_shape, s_bf_per_img in zip(scores, boxes, image_shapes, scores_bf_multiply)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

def fast_rcnn_inference_single_image_clip(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
    scores_bf_multiply: List[torch.Tensor],
    vis=False,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        scores_bf_multiply = scores_bf_multiply[valid_mask]

    scores = scores[:, :-1]
    scores_bf_multiply = scores_bf_multiply[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    scores_bf_multiply = scores_bf_multiply[filter_mask]

    # 2. Apply NMS for each class independently.
    if not soft_nms_enabled:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        keep, soft_nms_scores = batched_soft_nms(
            boxes,
            scores,
            filter_inds[:, 1],
            soft_nms_method,
            soft_nms_sigma,
            nms_thresh,
            soft_nms_prune,
        )
        scores[keep] = soft_nms_scores
        # scores_bf_multiply? (TBD)
        scores_bf_multiply = scores
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    scores_bf_multiply = scores_bf_multiply[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    if vis: # visualization: convert to the original scores before multiplying RPN scores
        result.scores = scores_bf_multiply
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)



def cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.cross_entropy(input, target, **kwargs)