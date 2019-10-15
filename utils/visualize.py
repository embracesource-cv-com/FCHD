import numpy as np
import cv2
import visdom
import time
import torch
from matplotlib import pyplot as plot


def show_origin_img(data):
    """
    Visualize the original image
    """
    path = data['img_path']
    boxes = data['boxes']

    img = cv2.imread(path)

    for box in boxes.astype(np.int):
        ymin, xmin, ymax, xmax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imshow('image', img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    # cv2.imwrite('raw_img.png', img)


def show_processed_img(img, boxes):
    """
    Visualize the pre-processed image
    """
    img = img.transpose((1, 2, 0))  # (C,H,W) -> (H,W,C)
    img_cp = img.copy()

    for box in boxes.astype(np.int):
        ymin, xmin, ymax, xmax = box
        cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imshow('image', img_cp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    # cv2.imwrite('precessed_img.png', img_cp)


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self._vis_kw = kwargs
        self.index = {}  # e.g.（’loss',23） the 23th value of loss
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',torch.Tensor(64,64))
        self.img('input_imgs',torch.Tensor(3,64,64))
        self.img('input_imgs',torch.Tensor(100,1,64,64))
        self.img('input_imgs',torch.Tensor(100,3,64,64),nrows=10)
        ！！！don‘torch ~~self.img('input_imgs',torch.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(torch.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self


def vis_image(img, ax=None):
    """Visualize a color image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox):
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=None)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

    return ax


def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it

    @param fig： a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plot.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(img, rois):
    fig = vis_bbox(img, rois)
    data = fig4vis(fig)
    return data
