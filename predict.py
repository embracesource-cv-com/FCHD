from networks.detector import HeadDetector
from skimage import transform
from data.preprocess import Normalize
from config import cfg
import torch
import numpy as np
import time
import cv2
import argparse


def preprocess(img, min_size=600, max_size=1000):
    c, h, w = img.shape
    scale = min(min_size / min(h, w), max_size / max(h, w))
    img = img / 255
    img = transform.resize(img, (c, h * scale, w * scale), mode='reflect')
    normalize = Normalize(mode='caffe')
    sample = normalize({'img': img})
    return sample['img'], scale


def read_img(path):
    img = cv2.imread(path)
    img_raw = img.copy()
    img = img.transpose((2, 0, 1))  # (H,W,C) -> (C,H,W)
    img, scale = preprocess(img)
    return img, img_raw, scale


def detect(img_path):
    # Load and pre-process img
    img, img_raw, scale = read_img(img_path)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.cuda().float()

    # Load model
    head_detector = HeadDetector(ratios=cfg.ANCHOR_RATIOS, scales=cfg.ANCHOR_SCALES)
    model_dict = torch.load(cfg.BEST_MODEL_PATH)['model']
    head_detector.load_state_dict(model_dict)
    head_detector = head_detector.cuda()

    # Inference
    begin = time.time()
    preds, scores = head_detector(img, scale, score_thresh=0.01)
    end = time.time()
    print("[INFO] Model inference time: {:.3f} s".format(end - begin))
    print("[PREDS SCORES]\n", scores)

    # Plot bbox into img
    for bbox in preds:
        ymin, xmin, ymax, xmax = bbox
        xmin, ymin = int(xmin / scale), int(ymin / scale)
        xmax, ymax = int(xmax / scale), int(ymax / scale)
        cv2.rectangle(img_raw, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imwrite('result.png', img_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img_path", help="path of the input image")
    args = parser.parse_args()
    detect(args.img_path)
