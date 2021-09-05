import h5py
import numpy as np

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        gt_label = fin['gt_label'][:]
        gt_mask = fin['gt_mask'][:]
        gt_valid = fin['gt_valid'][:]
        gt_other_mask = fin['gt_other_mask'][:]
        pts_rgb = fin['pts_rgb'][:]
        return pts, gt_label, gt_mask, gt_valid, gt_other_mask, pts_rgb

def load_data(fn):
    pts, gt_label, gt_mask, gt_valid, gt_other_mask, pts_rgb = load_h5(fn)
    return pts, gt_label, gt_mask, gt_valid, gt_other_mask, pts_rgb


pts, gt_label, gt_mask, gt_valid, gt_other_mask, pts_rgb = load_data('/data2/chengyang_zhao/baseline/partnet/v1/data/result/train/h5/train_Camera_102417_4.h5')
np.set_printoptions(threshold=np.inf)

print(pts[0].shape)