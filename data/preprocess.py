from skimage import transform
from torchvision import transforms
import torch
import numpy as np
from ..config import cfg


class Rescale(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, sample):
        img, boxes = sample['img'], sample['boxes']
        c, h, w = img.shape
        scale = min(self.min_size / min(h, w), self.max_size / max(h, w))
        img = img / 255
        img = transform.resize(img, (c, h * scale, w * scale), mode='reflect')
        new_boxes = boxes * scale
        return {'img': img, 'boxes': new_boxes, 'scale': scale}


class Normalize(object):
    def __init__(self, mode='caffe'):
        assert mode in ('caffe', 'torch'), 'Wrong mode.'
        self.mode = mode

    def __call__(self, sample):
        img = sample['img']

        if self.mode == 'caffe':
            img = img[[2, 1, 0], :, :]
            img = img * 255
            mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
            img -= mean
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            img = normalize(torch.from_numpy(img))
            img = img.numpy()
        sample['img'] = img

        return sample


def inverse_normalize(img):
    if cfg.CAFFE_PRETRAIN_PATH:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255
