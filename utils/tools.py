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


if __name__ == '__main__':
    print(generate_anchors(base_size=16, ratios=[0.5, 1., 2.], scales=[8, 16, 32]))
