import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.tools import generate_anchors


class RPN(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, ratios=(0.5, 1, 2), scales=(8, 16, 32),
                 feat_stride=16, proposal_creator_params=dict()):
        super(RPN, self).__init__()
        self.base_anchors = generate_anchors(16, ratios, scales)
        self.feat_stride = feat_stride

    def _shift(self, h, w):
        shift_x = np.arange(w) * self.feat_stride
        shift_y = np.arange(h) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel(), shift_y.ravel())).transpose()
        all_anchors = self.base_anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors
