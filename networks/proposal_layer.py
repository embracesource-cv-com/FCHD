from utils import tools
import numpy as np
from ..config import cfg


class ProposalLayer(object):
    def __init__(self, parent_model):
        self.parent_model = parent_model
        self.nms_thresh = cfg.RPN_NMS_THRESH
        self.min_size = cfg.RPN_MIN_SIZE

    def __call__(self, deltas, scores, anchors, img_size, scale):
        if self.parent_model.training:
            pre_nms_top_N = cfg.RPN_TRAIN_PRE_NMS_TOP_N
            post_nms_top_N = cfg.RPN_TRAIN_POST_NMS_TOP_N
        else:
            pre_nms_top_N = cfg.RPN_TEST_PRE_NMS_TOP_N
            post_nms_top_N = cfg.RPN_TEST_POST_NMS_TOP_N

        proposals = tools.bbox_regression(anchors, deltas)
        proposals = tools.clip_boxes(proposals, img_size)

        keep = tools.filter_boxes(proposals, self.min_size * scale)
        proposals = proposals[keep, :]
        scores = scores[keep]

        order = scores.argsort()[::-1]
        if pre_nms_top_N > 0:
            order = order[:pre_nms_top_N]
        proposals = proposals[order, :]
        scores = scores[order]

        keep = tools.nms(np.stack((proposals, scores.reshape(-1, 1))), self.nms_thresh)
        if post_nms_top_N > 0:
            keep = keep[:post_nms_top_N]
        rois = proposals[keep, :]
        scores = scores[keep]

        return rois, scores
