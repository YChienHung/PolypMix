import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class FinalConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(FinalConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class FinalConsistLoss(nn.Module):
    def __init__(self):
        super(FinalConsistLoss, self).__init__()

    def forward(self, patch_outputs, output):
        bs = output.shape[0]
        cls = output.shape[1]
        psz = patch_outputs.shape[-1]
        cn = output.shape[-1] // psz

        patch_outputs = patch_outputs.reshape(bs, cn, cn, cls, psz, psz)
        output = output.reshape(bs, cls, cn, psz, cn, psz).permute(0, 2, 4, 1, 3, 5)

        p_output_soft = torch.sigmoid(patch_outputs)
        outputs_soft = torch.sigmoid(output)

        loss = torch.mean((p_output_soft - outputs_soft) ** 2, dim=(0, 3, 4, 5)).sum()

        return loss


"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bce_loss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, reduction=reduction)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        loss = bce_loss + dice_loss

        return loss


""" Entropy Loss """


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt, labeled_bs):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4[:labeled_bs], gt[:labeled_bs])

    return loss0 + loss1 + loss2 + loss3 + loss4


def DeepSupervisionLoss_(pred, gt, labeled_bs):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0[:labeled_bs], gt[:labeled_bs])
    d1 = F.interpolate(d1[:labeled_bs], scale_factor=2, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gt[:labeled_bs])
    d2 = F.interpolate(d2[:labeled_bs], scale_factor=4, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt[:labeled_bs])
    d3 = F.interpolate(d3[:labeled_bs], scale_factor=8, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt[:labeled_bs])
    d4 = F.interpolate(d4[:labeled_bs], scale_factor=16, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gt[:labeled_bs])

    return loss0 + loss1 + loss2 + loss3 + loss4
