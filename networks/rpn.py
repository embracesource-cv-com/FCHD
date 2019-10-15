import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.tools import generate_anchors
from .proposal_layer import ProposalLayer
from config import cfg


class RPN(nn.Module):
    def __init__(self, c_in=512, c_out=512, ratios=(0.5, 1, 2), scales=(8, 16, 32),
                 feat_stride=16):
        super(RPN, self).__init__()
        self.base_anchors = generate_anchors(cfg.ANCHOR_BASE_SIZE, ratios, scales)
        self.feat_stride = feat_stride
        self.num_of_anchors = len(self.base_anchors)
        self.proposal_layer = ProposalLayer(self)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.cls_layer = nn.Conv2d(c_out, 2 * self.num_of_anchors, 1)
        self.regr_layer = nn.Conv2d(c_out, 4 * self.num_of_anchors, 1)
        self._weight_init()

    def forward(self, feature_map, img_size, scale):
        n, _, h, w = feature_map.size()
        anchors = self._shift(h, w)
        x = F.relu(self.conv1(feature_map))  # shape(n,512,30,40)

        rpn_regr = self.regr_layer(x)  # shape(n,8,30,40)
        rpn_regr = rpn_regr.permute(0, 2, 3, 1).reshape(n, -1, 4)  # shape(n,2400,4)

        rpn_cls = self.cls_layer(x)  # shape(n,4,30,40)
        rpn_cls = rpn_cls.permute(0, 2, 3, 1)  # shape(n,30,40,4)
        rpn_cls_fg = rpn_cls.reshape(n, h, w, self.num_of_anchors, 2)[:, :, :, :, 1]  # shape(n,30,40,2)
        rpn_cls_fg = rpn_cls_fg.reshape(n, -1)  # shape(n,2400)
        rpn_cls = rpn_cls.reshape(n, -1, 2)  # shape(n,2400,2)

        rois, rois_scores = self.proposal_layer(
            rpn_regr[0].detach().cpu().numpy(),
            rpn_cls_fg[0].detach().cpu().numpy(),
            anchors, img_size, scale
        )

        return rpn_regr, rpn_cls, rois, rois_scores, anchors

    def _shift(self, h, w):
        shift_x = np.arange(w) * self.feat_stride
        shift_y = np.arange(h) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel(), shift_x.ravel())).transpose()
        all_anchors = self.base_anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    def _weight_init(self, mean=0., std=0.01):
        """
        Initialize the weights of conv & cls & regr layer
        """
        for m in [self.conv1, self.cls_layer, self.regr_layer]:
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
