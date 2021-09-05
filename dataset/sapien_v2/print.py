import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

f = torch.load('/data2/helin/10.0_PointGroup/dataset/sapien_v2/train_Camera_101352_0_0.pth')
torch.set_printoptions(profile="full")
print(f[3].max())
