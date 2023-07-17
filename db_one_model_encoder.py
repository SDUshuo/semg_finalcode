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

number_of_vector_per_example =52
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
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

        self.autoencoder_tcn = Autoencoder(number_of_vector_per_example, encoding_dim).cuda()
        self.autoencoder_trn = Autoencoder(feature_bottleneck, encoding_dim).cuda()
        self.fenlei_feature =2*encoding_dim
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
        tcn_output = self.TCN(input_tcn)
        trn_output = self.TRN(input_trn)
        encoded_tcn, _ = self.autoencoder_tcn(tcn_output)
        encoded_trn, _ = self.autoencoder_trn(trn_output)
        fused_features = torch.cat((encoded_tcn, encoded_trn), dim=1)
        # fused_features = torch.add(encoded_tcn, encoded_trn)
        # Classification

        output = self.classifier(fused_features)
        return nn.functional.log_softmax(output)

        # print(lmf_output)
        # return lmf_output