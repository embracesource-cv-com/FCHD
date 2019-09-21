# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import HeadDataset
from data.preprocess import Rescale, Normalize
from config import cfg
from utils.visualize import check_raw_data, check_transformed_data
import torch
import numpy as np
import os
import cv2
import random

check_flag = True


def train():
    annots_path = os.path.join(cfg.DATASET_DIR, cfg.ANNOTS_FILE)
    transform = transforms.Compose([Rescale(), Normalize()])
    train_dataset = HeadDataset(cfg.DATASET_DIR, annots_path, transform)

    if check_flag:
        idx = random.randint(1, len(train_dataset))
        data = train_dataset.data_list[idx]
        check_raw_data(data)
        sample = train_dataset[idx]
        img, boxes = sample['img'], sample['boxes']
        check_transformed_data(img, boxes)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)


if __name__ == '__main__':
    train()
