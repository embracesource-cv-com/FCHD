import numpy as np
from utils import tools


class AnchorTargetLayer(object):
    def __init__(self, num_sample=32, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.num_sample = num_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, boxes, anchors, img_size):
        h, w = img_size
        total_anchors = len(anchors)

        # only keep anchors inside the image
        inds_inside = np.where(
            (anchors[:, 0] > 0) &
            (anchors[:, 1] > 0) &
            (anchors[:, 2] < h) &
            (anchors[:, 3] < w)
        )[0]
        anchors = anchors[inds_inside]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty(len(anchors), dtype=np.int32)
        labels.fill(-1)

        ious = tools.calc_ious(anchors, boxes)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(ious)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)

        labels[max_ious < self.neg_iou_thresh] = 0
        labels[gt_argmax_ious] = 1
        labels[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        num_pos = int(self.num_sample * self.pos_ratio)
        pos_inds = np.where(labels == 1)[0]
        if len(pos_inds) > num_pos:
            disable_inds = np.random.choice(pos_inds, len(pos_inds) - num_pos, replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_neg = self.num_sample - np.sum(labels == 1)
        neg_inds = np.where(labels == 0)[0]
        if len(neg_inds) > num_neg:
            disable_inds = np.random.choice(neg_inds, len(neg_inds) - num_neg, replace=False)
            labels[disable_inds] = -1

        targets = tools.bbox_transform(anchors, boxes[argmax_ious])
        targets = self._unmap(targets, total_anchors, inds_inside, fill=0)
        labels = self._unmap(labels, total_anchors, inds_inside, fill=-1)

        return targets, labels

    @staticmethod
    def _unmap(data, count, inds, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret
