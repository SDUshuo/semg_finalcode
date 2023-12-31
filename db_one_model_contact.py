import random

import numpy as np

import CNN_LSTM
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
from params_contact import fushion_dim ,rank,hidden_dim ,fushion_2_feature_bottleneck_,\
    num_channels,feature_bottleneck

number_of_vector_per_example =52
class Net(nn.Module):
    def __init__(self,tcn_inputs_channal,number_of_classes):
        super(Net, self).__init__()
        """定义各类模型 """
        # 创建TCN
        # 创建TCN
        self.lstm_out_dim = 52
        self.CNN_LSTM = CNN_LSTM.ConvLSTM(in_channels=tcn_inputs_channal, out_channels=self.lstm_out_dim,
                                          num_class=number_of_classes, fenlei=False).cuda()
        self.fushion_dim = fushion_dim #两个特征融合后的维度
        self.rank = rank
        self.hidden_dim = hidden_dim  #全连接层分类，前一层的维度
        # 这里的64指myTRN返回的64,不能随便改 也就是self.fushion_2_feature_bottleneck_
        # 创建 TRN模型
        self.TRN = myTRN.Net(number_of_class=number_of_classes, num_segments=12, fenlei=False).cuda()
        self.fenlei_feature =feature_bottleneck+self.lstm_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fenlei_feature, self.fenlei_feature),
            nn.BatchNorm1d(self.fenlei_feature),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fenlei_feature, number_of_classes)
        ).cuda()
        print("Number Parameters: DB1_finalemode", self.get_n_params())
    def get_n_params(self):
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            number_params = sum([np.prod(p.size()) for p in model_parameters])
            return number_params
    def forward(self, input_trn,input_tcn):
        trn_output = self.TRN(input_trn)

        lstm_output = self.CNN_LSTM(input_tcn)
        merged_data=torch.cat([trn_output,lstm_output],dim=-1)
        output = self.classifier(merged_data)
        return nn.functional.log_softmax(output)

        # print(lmf_output)
        # return lmf_output