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
import db_one_myTRN_attention as myTRN
import Wavelet_CNN_Source_Network
import TRNmodule
from params_dataaug import fushion_dim ,rank,hidden_dim ,fushion_2_feature_bottleneck_,\
    num_channels,feature_bottleneck
from final_eval_DB1 import window_len
number_of_vector_per_example =window_len
class Net(nn.Module):
    def __init__(self,tcn_inputs_channal,number_of_classes):
        super(Net, self).__init__()
        """定义各类模型 """
        # 创建TCN
        # 创建TCN
        self.TCN = TCNmodule.TemporalConvNet(num_inputs=tcn_inputs_channal, num_channels=num_channels , fc1_dim=number_of_vector_per_example,
                                        fc2_dim=number_of_vector_per_example, class_num=number_of_classes,
                                        kernel_size=3, dropout=0.3, fenlei=False).cuda()
        # self.Slowfushion=  Wavelet_CNN_Source_Network.Net(number_of_class=number_of_classes).cuda()
        # 创建LMF多模态融合
        self.fushion_dim = fushion_dim #两个特征融合后的维度
        self.rank = rank
        self.hidden_dim = hidden_dim  #全连接层分类，前一层的维度
        # 这里的64指myTRN返回的64,不能随便改 也就是self.fushion_2_feature_bottleneck_
        self.LMF = LMFmodule.LMF((feature_bottleneck, number_of_vector_per_example), self.fushion_dim, self.rank, self.hidden_dim, number_of_classes).cuda()
        # 创建 TRN模型
        self.TRN = myTRN.Net(number_of_class=number_of_classes, num_segments=12, fenlei=True).cuda()

        print("Number Parameters: DB1_finalemode", self.get_n_params())
    def get_n_params(self):
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            number_params = sum([np.prod(p.size()) for p in model_parameters])
            return number_params
    def forward(self, input_trn,input_tcn):
        # tcn_output = self.TCN(input_tcn)
        # trn_output = self.TRN(input_trn)
        # # slow_fushion_output =self.Slowfushion(input_trn)
        # # print("ssssssss")
        # # print(tcn_output.shape)
        # # print(trn_output.shape)
        # lmf_output = self.LMF(trn_output,tcn_output)
        # return lmf_output
        # print(lmf_output)

        trn_output = self.TRN(input_trn)
        return trn_output