# h5 files by chengyang zhao: n = 4
# n*N        pts = fin['pts'][:]
# n*N        gt_label = fin['gt_label'][:]
# n*100*N        gt_mask = fin['gt_mask'][:]
# n*100        gt_valid = fin['gt_valid'][:]
# n*N        gt_other_mask = fin['gt_other_mask'][:]
# n*N        pts_rgb = fin['pts_rgb'][:]

import h5py
import numpy as np
import torch

ROOT = '/data2/result/train/h5/'

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        gt_label = fin['gt_label'][:]
        gt_mask = fin['gt_mask'][:]
        gt_valid = fin['gt_valid'][:]
        gt_other_mask = fin['gt_other_mask'][:]
        pts_rgb = fin['pts_rgb'][:]
        return pts, gt_label, gt_mask, gt_valid, gt_other_mask, pts_rgb


def f(fn):
    dir = ROOT + fn
    pts, gt_label, gt_mask, gt_valid, gt_other_mask, pts_rgb = load_h5(dir)
    for i in range(pts.shape[0]):
        save_name = fn.strip('.h5')
        coords = pts[i] # - pts.mean(0)
        sem_labels = gt_label[i]
        colors = pts_rgb[i]
        instance_labels = np.zeros(gt_mask.shape[2])
        for j in range(gt_mask.shape[1]): # N
            instance_labels += gt_mask[i][j].astype(np.uint8) * (j+1) # TODO:?

        torch.save((coords, colors, sem_labels, instance_labels.astype(np.uint8)), save_name + f'_{i}.pth')
    # print(coords,'print\n', colors,'print\n', sem_labels,'print\n', instance_labels)
    print('Saving to ' + fn +f'_{i}.pth')


f('train_Camera_101352_0.h5')