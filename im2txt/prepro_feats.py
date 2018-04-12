#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import im_detect
from model.nms_wrapper import nms
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys, cv2, json

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

import numpy as np
import torch

if __name__ == "__main__":
    saved_model = saved_model = os.path.join('..', 'output', 'default.pth')
    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load model
    net = resnetv1(num_layers=101)
    net.create_architecture(81, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    net.load_state_dict(torch.load(saved_model))
    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    input_json = os.path.join('..', 'data', 'prepro', 'dataset_coco.json')
    if not os.path.isfile(input_json):
        raise IOError(('{:s} not found.').format(saved_model))

    imgs = json.load(open(input_json, 'r'))
    imgs = imgs['images']
    N = len(imgs)

    output_dir = os.path.join('..', 'data', 'feats')
    dir_det = output_dir + '_det'
    if not os.path.isdir(dir_det):
        os.mkdir(dir_det)

    images_root = os.path.join('..', 'data', 'coco', 'images')

    NMS_THRESH = 0.3
    conf_thresh = 0.2
    MIN_BOXES = 36
    MAX_BOXES = 36

    for i, img in enumerate(imgs):
        I = cv2.imread(os.path.join(images_root, img['filepath'], img['filename']))
        scores, boxes, fc7 = im_detect(net, I)

        max_conf = np.zeros((scores.shape[0]))
        for cls_ind in range(1, 81):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(torch.from_numpy(dets), NMS_THRESH)
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        fc7 = fc7[keep_boxes]
        # values = np.max(scores, axis=1)
        # inds = np.argsort(values)
        # fc7 = fc7[inds[:32], :]

        np.savez_compressed(os.path.join(dir_det, str(img['cocoid'])), feat=fc7.data.cpu().float().numpy())
        if i % 100 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    print('wrote ', output_dir)
