# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import HeadDataset
from data.preprocess import Rescale, Normalize, inverse_normalize
from config import cfg
from utils.visualize import visdom_bbox
import numpy as np
import os
from networks.detector import HeadDetector
from trainer import Trainer
from utils import tools
import time


def train():
    train_annots_path = os.path.join(cfg.DATASET_DIR, cfg.TRAIN_ANNOTS_FILE)
    val_annots_path = os.path.join(cfg.DATASET_DIR, cfg.VAL_ANNOTS_FILE)
    transform = transforms.Compose([Rescale(), Normalize()])
    train_dataset = HeadDataset(cfg.DATASET_DIR, train_annots_path, transform)
    val_dataset = HeadDataset(cfg.DATASET_DIR, val_annots_path, transform)

    print('[INFO] Load datasets.\n Training set size:{}, Verification set size:{}'
          .format(len(train_dataset), len(val_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    head_detector = HeadDetector(ratios=[1], scales=[2, 4])
    trainer = Trainer(head_detector).cuda()

    print('[INFO] Start training...')
    for epoch in range(cfg.EPOCHS):
        trainer.reset_meters()
        for i, data in enumerate(train_dataloader, 1):
            img, boxes, scale = data['img'], data['boxes'], data['scale']
            img, boxes = img.cuda().float(), boxes.cuda()
            scale = scale.item()

            trainer.train_step(img, boxes, scale)

            if i % cfg.PLOT_INTERVAL == 0:
                trainer.vis.plot_many(trainer.get_meter_data())
                origin_img = inverse_normalize(img[0].cpu().numpy())
                gt_img = visdom_bbox(origin_img, boxes[0].cpu().numpy())
                trainer.vis.img('gt_img', gt_img)
                preds, _ = trainer.head_detector.predict(img, scale)
                pred_img = visdom_bbox(origin_img, preds)
                trainer.vis.img('pred_img', pred_img)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')

        avg_accuracy = evaluate(val_dataloader, head_detector)

        print("[INFO] Epoch {} of {}.".format(epoch + 1, cfg.EPOCHS))
        print("\tValidate average accuracy: {:.3f}".format(avg_accuracy))

        time_str = time.strftime('%m%d%H%M')
        save_path = os.path.join(cfg.MODEL_DIR, 'checkpoint_{}_{:.3f}.pth'.format(time_str, avg_accuracy))
        trainer.save(save_path)
        if epoch == 8:
            trainer.load(save_path)
            trainer.scale_lr()


def evaluate(val_dataloader, head_detector):
    img_counts = 0
    accuracy = 0.0

    for data in val_dataloader:
        img, boxes, scale = data['img'], data['boxes'], data['scale']
        img, boxes = img.cuda().float(), boxes.cuda()
        scale = scale.item()

        preds, _ = head_detector.predict(img, scale)
        gts = boxes[0].cpu().numpy()
        if len(preds) == 0:
            img_counts += 1
        else:
            ious = tools.calc_ious(preds, gts)
            max_ious = ious.max(axis=1)
            correct_counts = len(np.where(max_ious >= 0.5)[0])
            gt_counts = len(gts)
            accuracy += correct_counts / gt_counts
            img_counts += 1

    avg_accuracy = accuracy / img_counts
    return avg_accuracy


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    train()
