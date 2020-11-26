import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from ..utils.conv2d_same import Conv2d as SameConv2d
from ..builder import BACKBONES, build_backbone

# SRM 层
class SRMFilter(nn.Module):

    def __init__(self):
        super(SRMFilter, self).__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=np.float32) / q[0]
        filter2 = np.asarray(filter2, dtype=np.float32) / q[1]
        filter3 = np.asarray(filter3, dtype=np.float32) / q[2]
        filters = np.asarray(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
        # 卷积核对于shape: out_dim, in_dum, ksize, ksize; tf 则是: ksize, ksize, in_dim, out_dim
        # self.weight = Parameter(torch.from_numpy(filters), requires_grad=False)
        self.filter = SameConv2d(in_channels=3, out_channels=3, kernel_size=5, bias=False, weight_=torch.from_numpy(filters))

    @torch.no_grad()
    def forward(self, x):
        # 都是用默认值, 使用 5x5 的卷积核, stride
        # return F.conv2d(x, weight=self.weight, stride=1, padding=2, dilation=1, groups=1)
        return self.filter(x)

