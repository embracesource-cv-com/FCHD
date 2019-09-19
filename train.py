# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import os
import cv2
from data.dataset import HeadDataset
from data.preprocess import Rescale, Normalize
from config import cfg


def train():
    annots_path = os.path.join(cfg.DATASET_DIR, cfg.ANNOTS_FILE)
    transform = transforms.Compose([Rescale(), Normalize])
    train_data = HeadDataset(cfg.DATASET_DIR, annots_path, transform)
    print(len(train_data))


if __name__ == '__main__':
    train()
