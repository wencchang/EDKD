import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def edkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
        * (temperature**2)
    )

    ###############################################
    import math
    alpha = 1.0
    beta = 8.0 * torch.exp(-1.0 * torch.tensor(1.0 - pred_teacher.max(1).values))
    # python tools/train.py --cfg configs/cifar100/edkd/res32x4_res8x4.yaml
    ###############################################

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return (alpha * tckd_loss).mean() + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class EDKD(Distiller):


    def __init__(self, student, teacher, cfg):
        super(EDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.EDKD.CE_WEIGHT
        self.alpha = cfg.EDKD.ALPHA
        self.beta = cfg.EDKD.BETA
        self.temperature = cfg.EDKD.T
        self.warmup = cfg.EDKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        p_s, p_t, beta = pred(logits_student, logits_teacher, target, self.temperature)
        loss_edkd = min(kwargs["epoch"] / self.warmup, 1.0) * edkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_edkd,
        }
        return logits_student, losses_dict, p_s, p_t, beta, target

def pred(logits_student, logits_teacher, target, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    p_t = pred_teacher.max(1).values
    p_s = pred_student.max(1).values
    beta = 8.0 * torch.exp(-1.0 * torch.tensor(1.0 - pred_teacher.max(1).values))

    return p_s, p_t, beta