# -*- coding: utf-8 -*-


class Config(object):
    DATASET_DIR = r'G:\fchd-dataset\brainwash'
    ANNOTS_FILE = 'brainwash_train.idl'
    CAFFE_PRETRAIN = True
    CAFFE_PRETRAIN_PATH = ''

    RPN_NMS_THRESH = 0.7,
    RPN_TRAIN_PRE_NMS_TOP_N = 12000,
    RPN_TRAIN_POST_NMS_TOP_N = 300,
    RPN_TEST_PRE_NMS_TOP_N = 6000,
    RPN_TEST_POST_NMS_TOP_N = 300,
    RPN_MIN_SIZE = 16


cfg = Config()
