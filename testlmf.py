import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

A = torch.randn(16, 512)
B = torch.randn(16, 1024)


n = A.shape[0]
A = torch.cat([A, torch.ones(n, 1)], dim=1)
B = torch.cat([B, torch.ones(n, 1)], dim=1)


# 假设所设秩: R = 4, 期望融合后的特征维度: h = 128
R, h = 4, 128
Wa = Parameter(torch.Tensor(R, A.shape[1], h))
Wb = Parameter(torch.Tensor(R, B.shape[1], h))

Wf = Parameter(torch.Tensor(1, R))
bias = Parameter(torch.Tensor(1, h))

# 分解后，并行提取各模态特征
fusion_A = torch.matmul(A, Wa)
fusion_B = torch.matmul(B, Wb)


# 利用一个Linear再进行特征融合（融合R维度）
funsion_ABC = fusion_A * fusion_B
funsion_ABC = torch.matmul(Wf, funsion_ABC.permute(1,0,2)).squeeze() + bias
print(funsion_ABC.shape)
