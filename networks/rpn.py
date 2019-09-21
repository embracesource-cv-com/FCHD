import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.tools import generate_anchors


class RPN(nn.Module):
    def __init__(self, c_in=512, c_out=512, ratios=(0.5, 1, 2), scales=(8, 16, 32),
                 feat_stride=16, proposal_creator_params=dict()):
        super(RPN, self).__init__()
        self.base_anchors = generate_anchors(16, ratios, scales)
        self.feat_stride = feat_stride
        self.num_of_anchors = len(self.base_anchors)
        self.proposal_layer = None  # todo: 生成 target anchor
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.cls_layer = nn.Conv2d(c_out, 2 * self.num_of_anchors, 1)
        self.regr_layer = nn.Conv2d(c_out, 4 * self.num_of_anchors, 1)

    def _shift(self, h, w):
        shift_x = np.arange(w) * self.feat_stride
        shift_y = np.arange(h) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel(), shift_y.ravel())).transpose()
        all_anchors = self.base_anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors
