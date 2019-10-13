# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import HeadDataset
from data.preprocess import Rescale, Normalize, inverse_normalize
from config import cfg
from utils.visualize import check_raw_data, check_transformed_data, visdom_bbox
import torch
import numpy as np
import os
import cv2
import random
from networks.detector import HeadDetector
from trainer import Trainer
from utils import tools


def train():
    train_annots_path = os.path.join(cfg.DATASET_DIR, cfg.TRAIN_ANNOTS_FILE)
    val_annots_path = os.path.join(cfg.DATASET_DIR, cfg.VAL_ANNOTS_FILE)
    transform = transforms.Compose([Rescale(), Normalize()])
    train_dataset = HeadDataset(cfg.DATASET_DIR, train_annots_path, transform)
    val_dataset = HeadDataset(cfg.DATASET_DIR, val_annots_path, transform)

    if cfg.DEBUG:
        idx = random.randint(1, len(train_dataset))
        data = train_dataset.data_list[idx]
        check_raw_data(data)
        sample = train_dataset[idx]
        img, boxes = sample['img'], sample['boxes']
        check_transformed_data(img, boxes)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    head_detector = HeadDetector(ratios=cfg.ANCHOR_RATIOS, scales=cfg.ANCHOR_SCALES)
    trainer = Trainer(head_detector).cuda()

    for epoch in range(cfg.EPOCHS):
        trainer.reset_meters()
        for i, data in enumerate(train_dataloader, 1):
            img, boxes, scale = data['img'], data['boxes'], data['scale']
            img, boxes = img.cuda(), boxes.cuda()

            if cfg.DEBUG:
                print('img size:', img.size())
                print('bboxes size:', boxes.size())
                print('scale:', scale)

            _, _, _ = trainer.train_step(img, boxes, scale)

            if i % cfg.PLOT_INTERVAL == 0:
                trainer.vis.plot_many(trainer.get_meter_data())
                origin_img = inverse_normalize(img[0].numpy())
                gt_img = visdom_bbox(origin_img, boxes[0].numpy())
                trainer.vis.img('gt_img', gt_img)
                preds, _ = trainer.head_detector.predict(img, scale)
                pred_img = visdom_bbox(origin_img, preds)
                trainer.vis.img('pred_img', pred_img)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')


def evaluate(val_dataloader, head_detector):
    """
    Given the dataloader of the test split compute the
    average corLoc of the dataset using the head detector
    model given as the argument to the function.
    """
    img_counts = 0
    correct_prob = 0.0
    for data in val_dataloader:
        img, boxes, scale = data['img'], data['boxes'], data['scale']
        img, boxes = img.cuda(), boxes.cuda()
        preds, _ = head_detector.predict(img, scale)
        gts = boxes[0]
        if len(preds) == 0:
            img_counts += 1
        else:
            ious = tools.calc_ious(preds, gts)
            max_ious = ious.max(axis=1)
            correct_counts = len(np.where(max_ious >= 0.5)[0])
            gt_counts = len(gts)
            correct_prob += correct_counts / gt_counts
            img_counts += 1
    return correct_prob / img_counts


if __name__ == '__main__':
    train()
