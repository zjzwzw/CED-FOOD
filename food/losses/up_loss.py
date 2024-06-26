
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from detectron2.utils.events import get_event_storage
import math
import numpy as np


class UpLoss(nn.Module):
    """Unknown Probability Loss
    """

    def __init__(self,
                 num_classes: int,
                 num_base_classes:int,
                 sampling_metric: str = "min_score",
                 sampling_ratio: int = 1,
                 topk: int = 3,
                 alpha: float = 1.0,
                 unk: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_base_classes = num_base_classes
        assert sampling_metric in ["min_score", "max_entropy", "random", "max_unknown_prob", "max_energy",
                                   "max_condition_energy", "VIM", "edl_dirichlet", "gradient"]
        self.sampling_metric = sampling_metric
        self.sampling_ratio = sampling_ratio
        # if topk==-1, sample len(fg)*2 examples
        self.topk = topk
        self.alpha = alpha
        self.unk = unk
        weight = torch.FloatTensor(1).fill_(0.1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        bias = torch.FloatTensor(1).fill_(0)
        self.bias = nn.Parameter(bias, requires_grad=True)


    def _soft_cross_entropy(self, input: Tensor, input_gt: Tensor, target: Tensor, targets_gt: Tensor, label, topk):

        #####
        # print("ratio",self.sampling_ratio)
        # print("k",self.topk)

        S_un = torch.sum(torch.exp(input) + 1, dim=1, keepdim=True)
        S_gt = torch.sum(torch.exp(input_gt) + 1, dim=1, keepdim=True)

        A_un = torch.sum(target * (torch.digamma(S_un) - torch.digamma(torch.exp(input) + 1,)), dim=1, keepdim=True)
        A_gt = torch.sum(targets_gt * (torch.digamma(S_gt) - torch.digamma(torch.exp(input_gt) + 1)), dim=1, keepdim=True)
        # A = torch.sum(y * (torch.log(S) - torch.log(torch.exp(scores) + 1)), dim=1, keepdim=True)
        A = A_un.sum() + A_gt.sum()
        return  A / input.shape[0]
        #####



    def _sampling(self, scores: Tensor, labels: Tensor, squarescores, objectness):
        fg_inds = labels != self.num_classes
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

        ###
        fg_square = squarescores[fg_inds]
        bg_square = squarescores[~fg_inds]
        fg_obj = objectness[fg_inds]
        bg_obj = objectness[~fg_inds]

        # remove unknown classes
        _fg_scores = fg_scores[:, :-2]
        _bg_scores = bg_scores[:, -1:]

        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg <
                                               self.topk) else self.topk
        # use maximum entropy as a metric for uncertainty
        # we select topk proposals with maximum entropy
        if self.sampling_metric == "max_entropy":
            pos_metric = dists.Categorical(
                _fg_scores.softmax(dim=1)).entropy()
            neg_metric = dists.Categorical(
                _bg_scores.softmax(dim=1)).entropy()
        # use minimum score as a metric for uncertainty
        # we select topk proposals with minimum max-score
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
            neg_metric = -_bg_scores.max(dim=1)[0]

        # we randomly select topk proposals
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0), ).to(scores.device)
            neg_metric = torch.rand(_bg_scores.size(0), ).to(scores.device)
        # max unknown prob
        elif self.sampling_metric == "max_unknown_prob":
            pos_metric = -fg_scores[:, -2]
            neg_metric = -bg_scores[:, -2]
        elif self.sampling_metric == "max_energy":
            pos_metric = -torch.logsumexp(fg_scores, dim=1)
            neg_metric = -torch.logsumexp(bg_scores, dim=1)
        elif self.sampling_metric == "edl_dirichlet":
            pos_metric = (self.num_classes + 1) / torch.sum(torch.exp(fg_scores) + 1, dim=1)
            neg_metric = (self.num_classes + 1) / torch.sum(torch.exp(bg_scores) + 1, dim=1)
        elif self.sampling_metric == "gradient":
            pos_metric = fg_square
            neg_metric = bg_square
        elif self.sampling_metric == "max_condition_energy":
            pos_metric = -torch.logsumexp(_fg_scores, dim=1)
            neg_metric = -torch.logsumexp(_bg_scores, dim=1)
        elif self.sampling_metric == "VIM":
            fg_scores_mean = fg_scores - torch.mean(fg_scores, dim=0)
            _fg_scores_mean_transpose = torch.transpose(fg_scores_mean, dim0=1, dim1=0)
            A = torch.mm(fg_scores_mean, _fg_scores_mean_transpose)
            # A_norm = torch.norm(A, p=2, dim=1).unsqueeze(1).expand_as(A)
            # A_normized = A.div(A_norm + 1e-5)
            A = A / (A.size()[0] - 1)
            (evals, evecs) = torch.eig(A, eigenvectors=True)
            evecs = evecs.detach()
            pos_metric = - evals[:, 0]
            _, pos_inds = pos_metric.topk(topk)
            R = evecs[:, pos_inds]
            R_transpose = torch.transpose(R, dim0=1, dim1=0)
            fg_scores_transform = torch.mm(R_transpose, fg_scores)
            fg_scores = fg_scores_transform
            fg_labels = fg_labels[pos_inds]
            neg_metric = -_bg_scores.max(dim=1)[0]

        if self.sampling_metric == "VIM":
            _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
            bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]
        else:
            _, pos_inds = pos_metric.topk(topk)
            _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
            bg_scores, bg_labels, bg_square, bg_obj = bg_scores[neg_inds], bg_labels[neg_inds], bg_square[neg_inds], bg_obj[neg_inds]
            fg_scores, fg_labels, fg_square, fg_obj = fg_scores[pos_inds], fg_labels[pos_inds], fg_square[pos_inds], fg_obj[pos_inds]
            # bg_scores, bg_labels, bg_ious = bg_scores[neg_inds], bg_labels[neg_inds], bg_ious[neg_inds]

            # aa = np.array(fg_ious.cpu())
            # bb = np.array(bg_ious.cpu())
            # with open("/home/subinyi/Users/FSOSOD/DeFRCN-main/fg_iou.txt", 'ab') as f:
            #     np.savetxt(f, aa)
            # with open("/home/subinyi/Users/FSOSOD/DeFRCN-main/bg_iou.txt", 'ab') as f:
            #     np.savetxt(f, bb)
        # _, pos_inds = pos_metric.topk(topk)
        # _, neg_inds = neg_metric.topk(topk*self.sampling_ratio)
        # fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
        # bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels, fg_square, bg_square, fg_obj, bg_obj, topk

    def forward(self, scores: Tensor, labels: Tensor, squarescores, objectness, ious):
        fg_scores, bg_scores, fg_labels, bg_labels, fg_squares, bg_squares, fg_obj, bg_obj, topk = self._sampling(
            scores, labels, squarescores, objectness)
        # sample both fg and bg
        # squares = torch.cat([fg_squares, bg_squares])
        scores = torch.cat([fg_scores, bg_scores])
        labels = torch.cat([fg_labels, bg_labels])
        # num_fg = fg_scores.size(0)

        final_cal_scores = scores[:, :-2]

        final_split_scores = torch.stack(final_cal_scores.split(self.num_classes-1, dim=1), dim=0)
        final_sum_scores = torch.sum(final_split_scores, dim=0)
        scores = torch.concat(
            (final_sum_scores, scores[:, -2:]), dim=1)






        num_sample, num_classes = scores.shape

        # un_scores = scores[:, -2]
        # print(un_scores)
        objs = torch.cat([fg_obj, bg_obj])
        # un_p = torch.sigmoid(un_scores)
        # kn_p = 1-un_p





        _, un_id = scores[:,-2:-1].max(dim=1)


        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != labels[:, None].repeat(1, num_classes)
        inds_un = mask != self.num_classes - 1
        mask_un = mask[inds_un].reshape(num_sample, num_classes-1)
        mask = mask[inds].reshape(num_sample, num_classes-1)

        # gt_scores = torch.gather(
        #     F.softmax(scores, dim=1), 1, labels[:, None]).squeeze(1)
        mask_scores_no_gt = torch.gather(scores, 1, mask)
        mask_scores_no_un = torch.gather(scores, 1, mask_un)
        # print(scores)
        # print(gt_scores)

        # gt_scores[gt_scores < 0] = 0.0
        targets = torch.zeros_like(mask_scores_no_gt)#Pu
        targets_gt = torch.zeros_like(mask_scores_no_un)
        num_fg = fg_scores.size(0)


        for i in range (0,num_fg):
            targets[i,self.num_classes-2+un_id[i]] = 1-objs[i]
            targets_gt[i, labels[i]] = objs[i]

        for i in range (num_fg,num_sample):
            targets[i,self.num_classes-1+un_id[i]] = objs[i]
            targets_gt[i,labels[i] - 1] = (1 - objs[i])*0.2


        return self._soft_cross_entropy(mask_scores_no_gt,mask_scores_no_un, targets.detach(),targets_gt.detach(), labels[0], topk)
