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
from params_contact import fushion_dim ,rank,hidden_dim ,fushion_2_feature_bottleneck_,\
    num_channels,feature_bottleneck,encoding_dim
import torch.nn.functional as F
number_of_vector_per_example =52
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))

    def forward(self, x):
        # x.shape: (batch_size, num_vectors, input_dim)
        attention_weights = torch.tanh(self.linear(x))  # shape: (batch_size, num_vectors, attention_dim)
        attention_weights = torch.matmul(attention_weights, self.context_vector)  # shape: (batch_size, num_vectors)
        attention_weights = F.softmax(attention_weights, dim=-1).unsqueeze(2)  # shape: (batch_size, num_vectors, 1)

        return torch.sum(x * attention_weights, dim=1)  # shape: (batch_size, input_dim)


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

        # 创建 TRN模型
        self.TRN = myTRN.Net(number_of_class=number_of_classes, num_segments=12, fenlei=False).cuda()

        # self.attention_tcn = Attention(number_of_vector_per_example, attention_dim_latefushion).cuda()
        # self.attention_trn = Attention(feature_bottleneck, attention_dim_latefushion).cuda()
        self.attention_dim =number_of_vector_per_example+feature_bottleneck
        self.attention_linear = nn.Linear(self.attention_dim, self.attention_dim).cuda()
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim),
            nn.BatchNorm1d(self.attention_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.attention_dim, number_of_classes)
        ).cuda()
        print("Number Parameters: DB1_finalemode", self.get_n_params())
    def get_n_params(self):
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            number_params = sum([np.prod(p.size()) for p in model_parameters])
            return number_params
    def forward(self, input_trn,input_tcn):
        tcn_output = self.TCN(input_tcn)
        trn_output = self.TRN(input_trn)
        # attention_tcn = self.attention_tcn(tcn_output.unsqueeze(1))
        # attention_trn = self.attention_trn(trn_output.unsqueeze(1))
        # fused_features = torch.cat((attention_tcn, attention_trn), dim=1)
        # Classification
        # Concatenate the two feature vectors along the dimension axis
        fused = torch.cat([tcn_output, trn_output], dim=1)  # shape: (batch_size, tcn_dim+trn_dim)

        # Calculate attention weights
        attention_weights = torch.nn.functional.softmax(self.attention_linear(fused),
                                                        dim=1)  # shape: (batch_size, tcn_dim+trn_dim)
        a= fused * attention_weights

        output = self.classifier(a)
        return nn.functional.log_softmax(output)

        # print(lmf_output)
        # return lmf_output