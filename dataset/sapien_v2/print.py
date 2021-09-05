import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

f = torch.load('/data2/helin/10.0_PointGroup/dataset/scannetv2/train/train_Camera_102829_1_2.pth')
torch.set_printoptions(profile="full")
print(f[2])
