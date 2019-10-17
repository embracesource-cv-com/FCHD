import torch


def rpn_regr_loss(pred_rger, gt_rger, gt_labels):
    """
    Calculate the regression loss of RPN

    :param pred_rger:
    :param gt_rger:
    :param gt_labels:
    :return:
    """
    weights = torch.zeros(gt_rger.size()).cuda()
    weights[(gt_labels > 0).view(-1, 1).expand_as(weights).cuda()] = 1
    loss = smooth_l1_loss(pred_rger, gt_rger, weights)
    loss /= (gt_labels >= 0).sum()  # ignore gt_labels == -1 (why not ignore labels=0?)
    return loss


def smooth_l1_loss(preds, gts, weights, sigma=3):
    sigma2 = sigma ** 2
    diff = (preds - gts) * weights
    abs_diff = diff.abs()
    flag = (abs_diff < (1 / sigma2)).float()
    loss = flag * (sigma2 / 2) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2)
    return loss.sum()
