import torch
from scipy import stats
import numpy as np

"""
The evaluation implementation refers to the following paper:
"Selective Feature Aggregation Network with Area-Boundary Constraints for Polyp Segmentation"
https://github.com/Yuqi-cuhk/Polyp-Seg
"""


def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = pred.round().float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = gt.round().float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # F2 score
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0

    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    target_flat = gt.view(size, -1)
    intersection = pred_flat * target_flat
    dice_score = torch.mean((2 * intersection.sum(1) + 1e-8) / (pred_flat.sum(1) + target_flat.sum(1) + 1e-8))

    return {
        'recall': Recall, 'specificity': Specificity, 'precision': Precision, 'F1': F1, 'Dice': dice_score, 'F2': F2,
        'ACC_overall': ACC_overall, 'IoU_poly': IoU_poly, 'IoU_bg': IoU_bg, 'IoU_mean': IoU_mean,
    }


class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = dict()
        for metric in metrics_list:
            self.metrics[metric] = list()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.metrics.keys():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.metrics[k].append(v)

    def mean(self):
        mean_metrics = dict()
        for k, v in self.metrics.items():
            mean_metrics[k] = np.mean(v)
        return mean_metrics

    def clean(self):
        for k in self.metrics.keys():
            self.metrics[k].clear()
