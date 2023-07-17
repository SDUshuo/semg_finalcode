import random

import numpy as np
import Wavelet_CNN_Source_Network as Wavelet_CNN_Source_Network
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode

import LMF as LMFmodule
import TCN as TCNmodule
import myTRN as myTRN
import TRNmodule
from save_data import number_of_vector_per_example,size_non_overlap,number_of_canals

class Net(nn.Module):
    def __init__(self,tcn_inputs_channal,number_of_classes):
        super(Net, self).__init__()
        """定义各类模型 """
        # 创建TCN
        self.TCN = TCNmodule.TemporalConvNet(num_inputs=tcn_inputs_channal, num_channels=[40, 15, 8, 1], fc1_dim=number_of_vector_per_example,
                                        fc2_dim=number_of_vector_per_example, class_num=number_of_classes,
                                        kernel_size=2, dropout=0.3, fenlei=False).cuda()
        # 创建LMF多模态融合
        fushion_dim = 80
        rank = 4
        hidden_dim = 64
        self.LMF = LMFmodule.LMF((64, number_of_vector_per_example), fushion_dim, rank, hidden_dim, number_of_classes).cuda()
        # 创建 TRN模型
        self.TRN = myTRN.Net(number_of_class=number_of_classes, num_segments=12, fenlei=False).cuda()

        print("Number Parameters: ", self.get_n_params())
    def get_n_params(self):
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            number_params = sum([np.prod(p.size()) for p in model_parameters])
            return number_params
    def forward(self, input_trn,input_tcn):
        tcn_output = self.TCN(input_tcn)
        trn_output = self.TRN(input_trn)
        lmf_output = self.LMF(trn_output,tcn_output)
        return lmf_output