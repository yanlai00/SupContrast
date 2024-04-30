"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # DEBUGGING
        min_correlation = torch.min(anchor_dot_contrast) * self.temperature
        # print("Minimum Correlation", min_correlation)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # import pdb; pdb.set_trace()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask * (1 - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + torch.exp(logits))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # print("SupCon Loss", loss)

        return loss

if __name__=='__main__':
    
    loss_fn = SupConLoss(temperature=0.1, base_temperature=0.1)

    batch_size = 9
    num_views = 2
    embd_dim = 128
    temperature = 0.1

    num_0 = 4
    num_1 = 3
    num_2 = 2

    features = torch.randn((batch_size, num_views, embd_dim))
    features = F.normalize(features, dim=2)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])

    loss_0 = 0
    for i0 in range(4):
        for view0 in range(num_views):
            zi = features[i0][view0]

            denom = 0
            for i0a in range(batch_size):
                for view0a in range(num_views):
                    if labels[i0a] != labels[i0]:
                        za = features[i0a][view0a]
                        denom += torch.exp(torch.dot(zi, za) / temperature)

            for i0p in range(4):
                for view0p in range(num_views):
                    if i0p != i0 or view0p != view0:
                        zp = features[i0p][view0p]
                        loss_0 += torch.log(torch.exp(torch.dot(zi, zp) / temperature) / (denom + torch.exp(torch.dot(zi, zp) / temperature)))

    loss_1 = 0
    for i0 in range(4, 7):
        for view0 in range(num_views):
            zi = features[i0][view0]

            denom = 0
            for i0a in range(batch_size):
                for view0a in range(num_views):
                    if labels[i0a] != labels[i0]:
                        za = features[i0a][view0a]
                        denom += torch.exp(torch.dot(zi, za) / temperature)

            for i0p in range(4, 7):
                for view0p in range(num_views):
                    if i0p != i0 or view0p != view0:
                        zp = features[i0p][view0p]
                        loss_1 += torch.log(torch.exp(torch.dot(zi, zp) / temperature) / (denom + torch.exp(torch.dot(zi, zp) / temperature)))

    loss_2 = 0
    for i0 in range(7, 9):
        for view0 in range(num_views):
            zi = features[i0][view0]

            denom = 0
            for i0a in range(batch_size):
                for view0a in range(num_views):
                    if labels[i0a] != labels[i0]:
                        za = features[i0a][view0a]
                        denom += torch.exp(torch.dot(zi, za) / temperature)

            for i0p in range(7, 9):
                for view0p in range(num_views):
                    if i0p != i0 or view0p != view0:
                        zp = features[i0p][view0p]
                        loss_2 += torch.log(torch.exp(torch.dot(zi, zp) / temperature) / (denom + torch.exp(torch.dot(zi, zp) / temperature)))

    gt_loss = (-1/7 * loss_0 -1/5 * loss_1 -1/3 * loss_2) / (batch_size * num_views)

    computed_loss = loss_fn(features, labels)

    print("Ground Truth", gt_loss)
    print("Computed", computed_loss)

    assert gt_loss - computed_loss < 1e-10

    