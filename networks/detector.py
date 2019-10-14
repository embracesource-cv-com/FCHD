import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from .base_net import BaseVGG16
from .rpn import RPN
from utils import tools


class HeadDetector(nn.Module):
    def __init__(self, ratios, scales):
        super(HeadDetector, self).__init__()
        self.extractor = BaseVGG16(caffe_pretrain=True)()
        self.rpn = RPN(ratios=ratios, scales=scales)
        self.lr = 0.001
        self.weight_decay = 0.0005
        self.use_adam = False

    def forward(self, x, scale):
        img_size = x.size()[2:]
        feature_map = self.extractor(x)
        rpn_regr, rpn_cls, rois, rois_scores, anchors = self.rpn(feature_map, img_size, scale)
        return rpn_regr, rpn_cls, rois, rois_scores, anchors

    def predict(self, x, scale, nms_thresh=0.3, score_thresh=0.01):
        h, w = x.size()[2:]
        _, _, rois, rois_scores, _ = self.forward(x, scale)
        rois[:, :4:2] = np.clip(rois[:, :4:2], 0, h)
        rois[:, 1:4:2] = np.clip(rois[:, 1:4:2], 0, w)

        probs = F.softmax(torch.from_numpy(rois_scores))
        probs = probs.numpy()

        mask = probs > score_thresh
        boxes = rois[mask]
        scores = probs[mask]

        keep = tools.nms(np.stack((boxes, scores.reshape(-1, 1))), nms_thresh)
        boxes = boxes[keep]
        scores = scores[keep]

        return boxes, scores

    def get_optimizer(self):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': self.lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': self.lr, 'weight_decay': self.weight_decay}]
        if self.use_adam:
            optimizer = optim.Adam(params)
        else:
            optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        return optimizer
