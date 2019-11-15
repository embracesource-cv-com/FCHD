# -*- coding: utf-8 -*-


class Config(object):
    DATASET_DIR = '/home/dataset/fchd_datas/brainwash'
    TRAIN_ANNOTS_FILE = 'brainwash_train.idl'
    VAL_ANNOTS_FILE = 'brainwash_val.idl'

    CAFFE_PRETRAIN = True
    CAFFE_PRETRAIN_PATH = './checkpoints/pre_trained/vgg16_caffe.pth'

    RPN_NMS_THRESH = 0.7
    RPN_TRAIN_PRE_NMS_TOP_N = 12000
    RPN_TRAIN_POST_NMS_TOP_N = 300
    RPN_TEST_PRE_NMS_TOP_N = 6000
    RPN_TEST_POST_NMS_TOP_N = 300
    RPN_MIN_SIZE = 16

    ANCHOR_BASE_SIZE = 16
    ANCHOR_RATIOS = [1]
    ANCHOR_SCALES = [2, 4]

    EPOCHS = 15
    PRINT_LOG = True
    PLOT_INTERVAL = 2
    VISDOM_ENV = 'fchd'

    MODEL_DIR = './checkpoints'
    BEST_MODEL_PATH = './checkpoints/checkpoint_best.pth'


cfg = Config()
