import torch
import torch.nn as nn
import torch.nn.functional as F


class my_model(nn.Module):
    def __init__(self, dims):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])

    # sigma is the standard deviation of the Gaussian noise
    def forward(self, x, is_train=True, sigma=0.01):
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        # dim=1: 表示对 行 进行归一化。p=2: 表示使用 L2 范数进行归一化。
        out1 = F.normalize(out1, dim=1, p=2)
        # 训练时，给out2加上高斯噪声
        if is_train:
            out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma)
        else:
            out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2
