# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

import numpy as np
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
from numpy.lib import pad
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import functional as F
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from tools.clip_prompt_utils import pre_tokenize, tokenize
from ..backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from PIL import Image
import copy
# from ..backbone.fpn import build_resnet_fpn_backbone
from ..backbone.clip_backbone import build_clip_language_encoder, ResidualAttentionBlock

from detectron2.layers.batch_norm import FrozenBatchNorm2d

__all__ = ["CLIPFastRCNN",]


class Grad_all_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()
        # self.feature = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()
        output.register_hook(_stor_grad)
        # self.feature = output.clone()

    def close(self):
        self.hook.remove()

@META_ARCH_REGISTRY.register()
class CLIPFastRCNN(nn.Module):
    """
    Fast R-CNN style where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components:
    1. Localization branch: pretrained backbone+RPN or equivalent modules, and is able to output object proposals
    2. Recognition branch: is able to recognize zero-shot regions
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
        language_encoder: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_c4: False,
        use_clip_attpool: False,
        offline_input_format: Optional[str] = None,
        offline_pixel_mean: Tuple[float],
        offline_pixel_std: Tuple[float],
        num_classes: int,
        is_base_train: bool,
        num_base_classes: int,
        concept_txt: str
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.offline_backbone = offline_backbone
        self.backbone = backbone
        self.lang_encoder = language_encoder
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False

        if self.training:
            bn_hooks = []
            # bn_hooks1 = []
            bn_hooks.append(Grad_all_hook(self.backbone.layer4[2].bn3))
            # for module in self.backbone.layer4.modules():
            #     if isinstance(module, FrozenBatchNorm2d):
            #         bn_hooks.append(Grad_all_hook(module))
            # for module in self.lang_encoder.transformer.modules():
            #     if isinstance(module, ResidualAttentionBlock):
            #         bn_hooks1.append(Grad_all_hook(module))
            self.hooks = bn_hooks
        # self.hooks = []

        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        self.num_classes = num_classes

        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        ####
        for p in self.lang_encoder.parameters():
            p.requires_grad = False

        concept_file = concept_txt
        # concept_feats = []
        classnames = []
        # self.token_emb= []
        with open(concept_file, 'r') as f:
            for line in f:
                concept = line.strip()
                classnames.append(concept)

        dtype = self.lang_encoder.dtype
        n_cls = self.num_classes-1
        n_ctx = 16
        ctx_dim = language_encoder.ln_final.weight.shape[0]
        self.base_cls_num = num_base_classes
        self.K = 1
        self.is_base_train = is_base_train

        ctx_vectors_base = torch.empty(self.base_cls_num, n_ctx, ctx_dim, dtype=dtype)  # base*16*512
        nn.init.normal_(ctx_vectors_base, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)  # X X X X...
        self.ctx_base = nn.Parameter(ctx_vectors_base)

        ###
        # if not is_base_train:
        #     self.ctx_base.requires_grad = False
        ###

        ctx_vectors_novel = torch.empty(self.num_classes-self.base_cls_num-1, n_ctx, ctx_dim, dtype=dtype)  # novel*16*512
        nn.init.normal_(ctx_vectors_novel, std=0.02)
        self.ctx_novel = nn.Parameter(ctx_vectors_novel)





        prompts = [prompt_prefix + " " + name + "." for name in classnames] * self.K


        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = language_encoder.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        self.judge = False


    @classmethod
    def from_config(cls, cfg):
        # create independent backbone & RPN
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN":
            # create offline cfg for the pretrained backbone & RPN
            from detectron2.config import get_cfg
            offline_cfg = get_cfg()
            offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
            # print(offline_cfg)
            if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
                offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
                offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
                offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
                offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9
            if cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST:
                offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000

            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg)
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        # region proposals are ground-truth boxes
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None

        backbone = build_backbone(cfg)
        # build language encoder
        if cfg.MODEL.CLIP.GET_CONCEPT_EMB: # extract concept embeddings
            language_encoder = build_clip_language_encoder(cfg)
        else:
            language_encoder = None
        language_encoder = build_clip_language_encoder(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())

        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn,
            "backbone": backbone,
            "language_encoder": language_encoder,
            "roi_heads": roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "is_base_train": cfg.MODEL.CLIP.BASE_TRAIN,
            "num_base_classes": cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES,
            "concept_txt": cfg.MODEL.CLIP.CONCEPT_TXT
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes",./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # localization branch: offline modules to get the region proposals
        with torch.no_grad():
            if self.clip_crop_region_type == "GT":  # from ground-truth
                proposals = []
                for r_i, b_input in enumerate(batched_inputs):
                    this_gt = copy.deepcopy(b_input["instances"])  # Instance
                    gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                    this_gt._fields = {'proposal_boxes': gt_boxes, 'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)}
                    proposals.append(this_gt)
            elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
                if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
                    self.offline_backbone.eval()
                    self.offline_proposal_generator.eval()
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # if not self.judge:
        #     tmp = self.ctx_base.clone().mean(dim=0)
        #     tmp = tmp.unsqueeze(0).expand(self.n_cls-self.base_cls_num,-1,-1)
        #     self.ctx_novel = nn.Parameter(tmp.clone())
        #     self.judge = True
        # if not self.judge:
        #     self.ctx_novel = nn.Parameter(self.ctx_base.clone())
        #     self.judge = True


        ####
        ctx_base = self.ctx_base
        # if ctx_base.dim() == 2:
        #     ctx_base = ctx_base.unsqueeze(0).expand(self.base_cls_num, -1, -1)
        ctx_novel = self.ctx_novel
        # ctx_novel = torch.cat([ctx_novel]*self.K, dim=0)
        # if ctx_novel.dim() == 2:
        #     ctx_novel = ctx_novel.unsqueeze(0).expand(self.n_cls - self.base_cls_num, -1, -1)
        # if self.is_base_train:
        #     ctx = ctx_base
        # else:
        #     ctx = torch.cat([ctx_base, ctx_novel]) #num_classes*16*512
        ctx = torch.cat([ctx_base, ctx_novel])
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )


        text_features = self.lang_encoder.encode_prompt(prompts, self.tokenized_prompts)



        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool, hooks = self.hooks, batched_input = batched_inputs, text_emb=text_features)
            # else: # use mean pool
            #     _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
        # else:  # regular detector setting
        #     if self.use_clip_attpool: # use att_pool from CLIP to match dimension
        #         _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, attnpool=self.backbone.bottom_up.attnpool)
        #     else: # use mean pool
        #         _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        # localization branch: offline modules to get the region proposals
        if self.clip_crop_region_type == "GT":  # from ground-truth
            proposals = []
            for r_i, b_input in enumerate(batched_inputs):
                this_gt = copy.deepcopy(b_input["instances"])  # Instance
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes} #, 'objectness_logits': None}
                proposals.append(this_gt)
        elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            if detected_instances is None:
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        ####
        ctx_base = self.ctx_base
        # if ctx_base.dim() == 2:
        #     ctx_base = ctx_base.unsqueeze(0).expand(self.base_cls_num, -1, -1)
        ctx_novel = self.ctx_novel
        # ctx_novel = torch.cat([ctx_novel] * self.K, dim=0)
        # if ctx_novel.dim() == 2:
        #     ctx_novel = ctx_novel.unsqueeze(0).expand(self.n_cls - self.base_cls_num, -1, -1)
        ctx = torch.cat([ctx_base, ctx_novel])
        # prefix = self.test_token_prefix
        # suffix = self.test_token_suffix
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        # ctx_un = self.ctx_unknown
        # un_prefix = self.token_un_prefix
        # un_suffix = self.token_un_suffix
        # un_prompt = torch.cat([un_prefix,ctx_un,un_suffix], dim=1)
        # prompts = torch.cat((prompts,un_prompt),dim=0)

        text_features = self.lang_encoder.encode_prompt(prompts, self.tokenized_prompts)

        # text_features = torch.cat([text_features[:self.K * self.base_cls_num],
        #                            torch.cat([text_features[self.K * self.base_cls_num:]] * self.K, dim=0)], dim=0)
        # g1, g2 = text_features[:self.K*self.base_cls_num], text_features[self.K*self.base_cls_num:]
        # g1_s, g2_s = torch.chunk(g1, self.K, dim=0), torch.chunk(g2, self.K, dim=0)
        #
        # text_features = torch.cat([torch.cat([g1_s[i],g2_s[i]],dim=0) for i in range(self.K)],dim=0)
        ####

        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4, attnpool=self.backbone.attnpool, text_emb=text_features)
            else: # use mean pool
                results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
        else:  # regular detector setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                results, _  = self.roi_heads(images, features, proposals, None, attnpool=self.backbone.bottom_up.attnpool)
            else:
                results, _  = self.roi_heads(images, features, proposals, None)

        #visualize_proposals(batched_inputs, proposals, self.input_format)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPFastRCNN._postprocess(results, batched_inputs)
        else:
            return results

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

