import torch
import torch.nn as nn
from torch.nn import functional as F
from networks.anchor_target_layer import AnchorTargetLayer
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils import losses
from collections import namedtuple
from utils.visualize import Visualizer
from config import cfg


class Trainer(nn.Module):
    def __init__(self, head_detector):
        super(Trainer, self).__init__()
        self.head_detector = head_detector
        self.optimizer = self.head_detector.get_optimizer()
        self.anchor_target_layer = AnchorTargetLayer()
        self.loss_tuple = namedtuple('LossTuple',
                                     ['rpn_regr_loss',
                                      'rpn_cls_loss',
                                      'total_loss'])
        self.vis = Visualizer(env=cfg.VISDOM_ENV)
        self.rpn_cm = ConfusionMeter(2)  # confusion matrix with 2 classes
        self.meters = {k: AverageValueMeter() for k in self.loss_tuple._fields}  # average loss

    def forward(self, x, gt_boxes, scale):
        batch = x.size()[0]
        assert batch == 1, 'Currently only batch size 1 is supported.'
        img_size = x.size()[2:]

        # Forward pass
        feature_map = self.head_detector.extractor(x)
        rpn_regr, rpn_cls, _, _, anchors = self.head_detector.rpn(feature_map, img_size, scale)

        # Remove the batch dimension
        gt_boxes, rpn_regr, rpn_cls = gt_boxes[0], rpn_regr[0], rpn_cls[0]

        # Generates GT regression targets and GT labels
        gt_regr, gt_cls = self.anchor_target_layer(gt_boxes.numpy(), anchors, img_size)
        gt_regr = torch.from_numpy(gt_regr).cuda().float()
        gt_cls = torch.from_numpy(gt_cls).cuda().long()

        # Computes loss
        rpn_regr_loss = losses.rpn_regr_loss(rpn_regr, gt_regr, gt_cls)
        rpn_cls_loss = F.cross_entropy(rpn_cls, gt_cls, ignore_index=-1)
        total_loss = rpn_regr_loss + rpn_cls_loss
        loss_list = [rpn_regr_loss, rpn_cls_loss, total_loss]

        # Ignore samples with a label = -1
        valid_gt_cls = gt_cls[gt_cls > -1]
        valid_pred_cls = rpn_cls[gt_cls > -1]

        # Computes the confusion matrix
        self.rpn_cm.add(valid_pred_cls.detach(), valid_gt_cls.detach())

        return self.loss_tuple(*loss_list)

    def train_step(self, x, boxes, scale):
        loss_tuple = self.forward(x, boxes, scale)
        self.optimizer.zero_grad()
        loss_tuple.total_loss.backward()
        self.optimizer.step()
        self.update_meters(loss_tuple)

    def update_meters(self, loss_tuple):
        loss_dict = {k: v.item() for k, v in loss_tuple._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_dict[key])

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def save(self, path, save_optimizer=False):
        save_dict = dict()
        save_dict['model'] = self.head_detector.state_dict()
        save_dict['vis_info'] = self.vis.state_dict()
        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        torch.save(save_dict, path)
        self.vis.save([self.vis.env])

    def load(self, path, load_optimizer=True):
        state_dict = torch.load(path)
        self.head_detector.load_state_dict(state_dict['model'])
        if load_optimizer and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
