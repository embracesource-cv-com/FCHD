import numpy as np


def generate_anchors(base_size, ratios, scales):
    """
    generate base anchors by enumerating aspect ratios and scales

    :param base_size: int, the size of the reference window.
    :param ratios: list of float, e.g. [0.5, 1., 2.]
    :param scales: list of int, e.g [8, 16, 32]
    :return: nd-array, shape(size of scales * size of ratios, 4)
    """
    scales = np.expand_dims(np.array(scales), axis=0)
    ratios = np.expand_dims(np.array(ratios), axis=1)

    hs = base_size * scales * np.sqrt(ratios)
    ws = base_size * scales * np.sqrt(1 / ratios)
    hs = hs.reshape(-1, 1)
    ws = ws.reshape(-1, 1)

    ctr_x = ctr_y = 0.5 * (base_size - 1)
    # the format is [y1, x1, y2, x2]
    anchors = np.hstack((ctr_y - 0.5 * (hs - 1),
                         ctr_x - 0.5 * (ws - 1),
                         ctr_y + 0.5 * (hs - 1),
                         ctr_x + 0.5 * (ws - 1),))
    return np.round(anchors)


def bbox_regression(boxes, deltas):
    hs = boxes[:, 2] - boxes[:, 0]
    ws = boxes[:, 3] - boxes[:, 1]
    ctr_y = boxes[:, 0] + 0.5 * hs
    ctr_x = boxes[:, 1] + 0.5 * ws

    dy = deltas[:, :1]
    dx = deltas[:, 1:2]
    dh = deltas[:, 2:3]
    dw = deltas[:, 3:]

    pred_ctr_y = dy * hs[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_x = dx * ws[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_h = np.exp(dh) * hs[:, np.newaxis]
    pred_w = np.exp(dw) * ws[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape)
    pred_boxes[:, :1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 1:2] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 2:3] = pred_ctr_y + 0.5 * pred_h  # y2
    pred_boxes[:, 3:] = pred_ctr_x + 0.5 * pred_w  # x2

    return pred_boxes


def bbox_transform(anchor_boxes, gt_boxes):
    anchor_hs = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_ws = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    anchor_ctr_y = anchor_boxes[:, 0] + 0.5 * anchor_hs
    anchor_ctr_x = anchor_boxes[:, 1] + 0.5 * anchor_ws

    gt_hs = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_ws = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_y = gt_boxes[:, 0] + 0.5 * gt_hs
    gt_ctr_x = gt_boxes[:, 1] + 0.5 * gt_ws

    dy = (gt_ctr_y - anchor_ctr_y) / anchor_hs
    dx = (gt_ctr_x - anchor_ctr_x) / anchor_ws
    dh = np.log(gt_hs / anchor_hs)
    dw = np.log(gt_ws / anchor_ws)

    targets = np.vstack((dy, dx, dh, dw)).transpose()

    return targets


def clip_boxes(boxes, img_size):
    """
    Clip boxes to image boundaries.
    """
    #  0 =< y1,y2 <= h
    boxes[:, :4:2] = np.clip(boxes[:, :4:2], 0, img_size[0] - 1)
    #  0 =< x1,x2 <= w
    boxes[:, 1:4:2] = np.clip(boxes[:, 1:4:2], 0, img_size[1] - 1)

    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    hs = boxes[:, 2] - boxes[:, 0] + 1
    ws = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def nms(boxes, thresh):
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(boxes.shape[0])

    keep = []
    while order.size() > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        ws = np.maximum(0, xx2 - xx1 + 1)
        hs = np.maximum(0, yy2 - yy1 + 1)
        inter = ws * hs
        ious = inter / (areas[i] + areas[order] - inter)

        inds = np.where(ious <= thresh)[0]
        order = order[inds]
        keep.append(i)

    return keep


def calc_ious(boxes_a, boxes_b):
    """
    Calculate IOUs between boxes a and boxes b
    :param boxes_a: numpy array, shape(N, 4)
    :param boxes_b: numpy array, shape(M, 4)
    :return: numpy array, shape(N, M)
    """
    boxes_a = np.expand_dims(boxes_a, axis=1)  # (N, 1, 4)
    boxes_b = np.expand_dims(boxes_b, axis=0)  # (1, M, 4)

    areas_a = (boxes_a[..., 2] - boxes_a[..., 0] + 1) * (boxes_a[..., 3] - boxes_a[..., 1] + 1)  # (N, 1)
    areas_b = (boxes_b[..., 2] - boxes_b[..., 0] + 1) * (boxes_b[..., 3] - boxes_b[..., 1] + 1)  # (1, M)

    xx1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])  # (N, M)
    yy1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    xx2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])
    yy2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])

    ws = np.maximum(0, xx2 - xx1 + 1)
    hs = np.maximum(0, yy2 - yy1 + 1)
    inters = ws * hs

    ious = inters / (areas_a + areas_b - inters)
    return ious


if __name__ == '__main__':
    print(generate_anchors(base_size=16, ratios=[0.5, 1., 2.], scales=[8, 16, 32]))
