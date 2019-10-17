import numpy as np
from utils import tools


class AnchorTargetLayer(object):
    """
    Generates GT regression targets and GT classification labels for each anchor.
    """
    def __init__(self, num_sample=32, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.num_sample = num_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, gt_boxes, anchors, img_size):
        h, w = img_size
        num_anchors = len(anchors)

        # Only keep anchors inside the image
        inds_inside = np.where(
            (anchors[:, 0] > 0) &
            (anchors[:, 1] > 0) &
            (anchors[:, 2] < h) &
            (anchors[:, 3] < w)
        )[0]
        anchors = anchors[inds_inside]

        # Store the label of each anchor (1 is positive, 0 is negative, -1 is ignored)
        labels = np.empty(len(anchors), dtype=np.int32)
        labels.fill(-1)

        ious = tools.calc_ious(anchors, gt_boxes)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(ious)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)

        # Label allocation criteria:
        # 1.iou is less than 0.3 is negative,
        # 2.iou is not less than 0.7 is positive,
        # 3.anchors with the largest iou of GTs is positive.
        labels[max_ious < self.neg_iou_thresh] = 0
        labels[max_ious >= self.pos_iou_thresh] = 1
        labels[gt_argmax_ious] = 1

        # Subsample positive labels if have too many
        num_pos = int(self.num_sample * self.pos_ratio)
        pos_inds = np.where(labels == 1)[0]
        if len(pos_inds) > num_pos:
            disable_inds = np.random.choice(pos_inds, len(pos_inds) - num_pos, replace=False)
            labels[disable_inds] = -1

        # Subsample negative labels if have too many
        num_neg = self.num_sample - np.sum(labels == 1)
        neg_inds = np.where(labels == 0)[0]
        if len(neg_inds) > num_neg:
            disable_inds = np.random.choice(neg_inds, len(neg_inds) - num_neg, replace=False)
            labels[disable_inds] = -1

        # Generates regression targets
        targets = tools.bbox_transform(anchors, gt_boxes[argmax_ious])

        # Map back to the original size
        targets = self._unmap(targets, num_anchors, inds_inside, fill=0)
        labels = self._unmap(labels, num_anchors, inds_inside, fill=-1)

        return targets, labels

    @staticmethod
    def _unmap(data, count, inds, fill):
        """
        Map the subset (data) back to the original set size
        """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=data.dtype)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[inds, :] = data
        return ret
