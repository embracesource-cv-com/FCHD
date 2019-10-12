import numpy as np
from PIL import Image, ImageDraw
import cv2
import visdom
import time
import torch


def draw_box(img, box, color='red', thickness=4, use_normalized_coordinates=False):
    img_pil = Image.fromarray(img).convert('RGB')
    _draw(img_pil, box, color, thickness, use_normalized_coordinates)
    np.copyto(img, np.array(img_pil))


def _draw(img, box, color, thickness, use_normalized_coordinates):
    ymin, xmin, ymax, xmax = box
    draw = ImageDraw.Draw(img)
    im_width, im_height = img.size
    if use_normalized_coordinates:
        left, right, top, bottom = xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom),
              (right, top), (left, top)], width=thickness, fill=color)


def check_raw_data(data_info):
    path = data_info['img_path']
    counts = data_info['counts']
    boxes = data_info['boxes']
    img = cv2.imread(path)
    img_cp = np.copy(np.asarray(img, dtype=np.uint8))
    for i in range(counts):
        draw_box(img_cp, boxes[i])

    cv2.imshow('image', img_cp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def check_transformed_data(img, boxes):
    img = img.transpose((1, 2, 0))
    img_cp = np.copy(np.asarray(img, dtype=np.uint8))
    for i in range(len(boxes)):
        draw_box(img_cp, boxes[i])

    cv2.imshow('image', img_cp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


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
