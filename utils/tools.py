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

    x_ctr = y_ctr = 0.5 * (base_size - 1)
    # the format is [ymin, xmin, ymax, xmax]
    anchors = np.hstack((y_ctr - 0.5 * (hs - 1),
                         x_ctr - 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),))
    return np.round(anchors)


if __name__ == '__main__':
    print(generate_anchors(base_size=16, ratios=[0.5, 1., 2.], scales=[8, 16, 32]))
