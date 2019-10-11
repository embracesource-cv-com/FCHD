import torch
import torch.nn as nn
from torch.nn import functional as F
from networks.anchor_target_layer import AnchorTargetLayer
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils import losses
from collections import namedtuple


class Trainer(nn.Module):
    def __init__(self, head_detector):
        super(Trainer, self).__init__()
        self.head_detector = head_detector
        self.optimizer = self.head_detector.get_optimizer()
        self.anchor_target_layer = AnchorTargetLayer()
        self.loss_tuple = namedtuple('LossTuple',
                                     ['rpn_regr_loss',
                                      'rpn_cls_loss',
                                      'total_loss'])
        self.vis = None  # todo:
        self.rpn_cm = ConfusionMeter(2)
        self.meters = {k: AverageValueMeter() for k in self.loss_tuple._fields}  # average loss

    def forward(self, x, boxes, scale):
        batch = x.size()[0]
        if batch != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        img_size = x.size()[2:]
        feature_map = self.head_detector.extractor(x)
        rpn_regr, rpn_cls, rois, rois_scores, anchors = self.head_detector.rpn(feature_map, img_size, scale)
        boxes, rpn_regr, rpn_cls = boxes[0], rpn_regr[0], rpn_cls[0]

        gt_rpn_regr, gt_rpn_cls = self.anchor_target_layer(boxes, anchors, img_size)
        gt_rpn_regr = torch.from_numpy(gt_rpn_regr).cuda()
        gt_rpn_cls = torch.from_numpy(gt_rpn_cls).cuda()
        
        rpn_regr_loss = losses.rpn_regr_loss(rpn_regr, gt_rpn_regr, gt_rpn_cls)
        rpn_cls_loss = F.cross_entropy(rpn_cls, gt_rpn_cls, ignore_index=-1)
        total_loss = rpn_regr_loss + rpn_cls_loss
        loss_list = [rpn_regr_loss, rpn_cls_loss, total_loss]

        valid_gt_cls = gt_rpn_cls[gt_rpn_cls > -1]
        valid_pred_cls = rpn_cls[gt_rpn_cls > -1]
        self.rpn_cm.add(valid_pred_cls, valid_gt_cls)

        return self.loss_tuple(*loss_list), rois, rois_scores

    def train_step(self, imgs, bboxes, scale):
        pass
