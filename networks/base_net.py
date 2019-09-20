import torch
from torch import nn
from torchvision.models import vgg16
from config import cfg


class BaseVGG16(object):
    def __init__(self, caffe_pretrain=False):
        self.caffe_pretrain = caffe_pretrain

    def __call__(self, *args, **kwargs):
        if self.caffe_pretrain:
            # load the caffe model
            model = vgg16(pretrained=False)
            model.load_state_dict(torch.load(cfg.CAFFE_PRETRAIN_PATH))
        else:
            # load the default torch model
            model = vgg16(pretrained=True)

        features = list(model.features)[:30]
        return nn.Sequential(*features)
