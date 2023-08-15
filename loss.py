import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def inter_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss


def intra_contrastive_loss(hs, hs_aug_list):
    sim_list = []
    positive_list = torch.Tensor().to(hs.device)
    temperature_f = 0.08
    k, _ = hs.size()
    for hs_agu in hs_aug_list:
        h = torch.cat((hs, hs_agu), dim=0)
        sim = torch.matmul(h, h.T) / temperature_f
        positive_samples = torch.diag(sim, k).reshape(k, 1)

        sim_list.append(sim)
        positive_list = torch.cat([positive_list, positive_samples], dim=1)

    mask = getmask(k)
    negative_samples = sim_list[0][:k, :k][mask].reshape(k, -1)
    logits = torch.cat((positive_list, negative_samples), dim=1)

    pos_label = torch.Tensor([1 for _ in range(positive_list.size(1))]).to(positive_list.device).long()
    neg_label = torch.Tensor([0 for _ in range(negative_samples.size(1))]).to(negative_samples.device).long()
    labels = torch.cat([pos_label, neg_label])

    l = 0.
    for logit in logits:
        loss_ = -(F.log_softmax(logit, dim=0) * labels).sum() / labels.sum()
        l += loss_
    l /= k
    return l


def getmask(k):
    ones = torch.ones((k, k))
    mask = ones.fill_diagonal_(0)
    mask = mask.bool()
    return mask
