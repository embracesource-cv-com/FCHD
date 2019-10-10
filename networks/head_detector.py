import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .base_net import BaseVGG16
from .rpn import RPN


class HeadDetector(nn.Module):
    def __init__(self, ratios, scales):
        super(HeadDetector, self).__init__()
        self.extractor = BaseVGG16(caffe_pretrain=True)
        self.rpn = RPN(ratios=ratios, scales=scales)
        self.lr = 0.001
        self.weight_decay = 0.0005

    def forward(self, x, scale):
        img_size = x.size()[2:]
        feature_map = self.extractor(x)
        rpn_regr, rpn_cls, rois, rois_scores, anchors = self.rpn(feature_map, img_size, scale)
        return rpn_regr, rpn_cls, rois, rois_scores, anchors

    def get_optimizer(self):
        pass
