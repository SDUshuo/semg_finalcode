import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from McDropout import McDropout
import numpy as np
# import home.yanshuo.YS.FinalCode.PyTorchImplementation.CWT.model.TRNmodule as TRNmodule
import torchvision.models as models
# import PyTorchImplementation.CWT.model.TRNmodule as TRNmodule
import TRNmodule
# from TRNmodule import return_TRN
from params_contact import fushion_2_feature_bottleneck_,feature_bottleneck,attention_dim
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
        weighted_vectors = x * attention_weights  # shape: (batch_size, num_vectors, input_dim)
        return weighted_vectors  # shape: (batch_size, num_vectors, input_dim)
        # Apply the attention weights
        # weighted_vectors = x * attention_weights  # shape: (batch_size, num_vectors, input_dim)
        #
        # # Concatenate the vectors
        # concatenated = weighted_vectors.view(weighted_vectors.shape[0],
        #                                      -1)  # shape: (batch_size, num_vectors * input_dim)
        # return concatenated
class Net(nn.Module):
    def __init__(self, number_of_class,num_segments=12, fenlei=True):
        super(Net, self).__init__()
        self.fenlei=fenlei
        self._input_batch_norm = nn.BatchNorm2d(num_segments, eps=1e-4)
        #self._input_prelu = pelu((1, 12, 1, 1))
        self._input_prelu = nn.PReLU(num_segments)
        self.all_first_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3),
            # nn.BatchNorm2d(2),
            # nn.ReLU(),
            # nn.Conv2d(2, 2, kernel_size=3),
            # nn.BatchNorm2d(2),
            # nn.ReLU(),
            # nn.Dropout()
        )
        # 创建一个虚拟数据点并将其传递给卷积层来获得其输出尺寸
        x = torch.randn(1, 1, *[10,7])
        self.img_feature_dim = self.num_flat_features(self.all_first_conv(x))
        # self.all_first_conv=nn.Conv2d(1, 1, kernel_size=2)
        """修改了第一层卷积，12个分别卷积"""
        # self.all_first_conv = []
        # for i in range(12):
        #     self.all_first_conv.append(nn.Conv2d(1, 1, kernel_size=3).cuda())

        #这个要和conv_feature=self.all_first_conv(x[:, i, :, :].unsqueeze(1)).cuda().view(x.shape[0], -1) 搭配使用
        # self.img_feature_dim=216
        self.num_segments=num_segments #有多少帧
        self.feature_bottleneck = feature_bottleneck
        self.scale=6
        self.trn = TRNmodule.return_TRN('TRNmultiscale', self.img_feature_dim, self.scale, number_of_class, self.feature_bottleneck)
        self.num_class=number_of_class
        self.input_dim = self.img_feature_dim

        self.attention = Attention( self.input_dim, attention_dim).cuda()
        self.first_part_FC=[]
        for i in range(2):
            self.first_part_FC.append(nn.Sequential(
                nn.Linear(self.feature_bottleneck, self.feature_bottleneck),
                # nn.BatchNorm1d(self.feature_bottleneck),
                nn.ReLU(),
                nn.Dropout()
            ).cuda())
        # for i in range(4):
        #     self.first_part_FC.append(nn.Sequential(
        #         nn.Linear(self.feature_bottleneck, self.feature_bottleneck),
        #         # nn.BatchNorm1d(self.feature_bottleneck),
        #         nn.ReLU(),
        #         nn.Dropout()
        #     ).cuda())
        # self.second_part_FC = []
        # self.fushion_2_feature_bottleneck_=fushion_2_feature_bottleneck_
        # for i in range(2):
        #     self.second_part_FC.append(nn.Sequential(
        #         nn.Linear(self.feature_bottleneck, self.feature_bottleneck),
        #         # nn.BatchNorm1d(self.fushion_2_feature_bottleneck_),
        #         nn.ReLU(),
        #         nn.Dropout()
        #     ).cuda())

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_bottleneck, self.feature_bottleneck),
            nn.BatchNorm1d(self.feature_bottleneck),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_bottleneck, self.num_class)
        )
        # self.classifier = nn.Linear(self.fushion_2_feature_bottleneck_, self.num_class)
        # print(self)
        #
        # print("Number Parameters: ", self.get_n_params())
    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批处理维度外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        """
        x：[batchsize,12,10,7]
        对数据处理，对每个批次包含的12帧的图片化成12个向量，送入trn，返回[batchsize,dim]
        """

        x = self._input_prelu(self._input_batch_norm(x))
        # print(x.shape)
        all_x_flattend=[]
        all_att_data=None
        #12帧图片
        for i in range(self.num_segments):
            #[(batchsize,feature_dim),]
            # x[:, i, :, :].unsqueeze(1).shape: torch.Size([batchsize, 1, 8, 7])
            conv_feature=self.all_first_conv(x[:, i, :, :].unsqueeze(1)).cuda().view(x.shape[0], -1)#torch.Size([batchsize, 40])  [512, 216]
            all_x_flattend.append(conv_feature)
            att_conv_feature = torch.stack(all_x_flattend, dim=1)
            all_att_data=self.attention(att_conv_feature)

        batch_size, dim, feature_size = all_att_data.size()  # 获取张量的形状
        # 使用 torch.split 分割张量
        chunks = torch.split(all_att_data, self.scale, dim=1)  # chunks 是一个元组

        # 如果你希望得到一个列表，可以这样做
        chunks_list = list(chunks)  # 将元组转换为列表

        #将12帧分为4组，每组3帧

        fenzu_trn_data=[]
        #每组3帧图片送入trn中，输入维度[batchsize,3,feature_dim]  如果12帧，那就是三组
        for fenzu_data in chunks_list:
            #fenzu_data 列表 4个tensor
            # combined_tensor = torch.stack(fenzu_data, dim=0).transpose(0, 1) #torch.Size([batchsize, scale, 40])

            fenzu_trn_data.append(self.trn( fenzu_data))

        first_merge_first_branch_data = fenzu_trn_data[0].clone()
        # print(first_merge_first_branch_data.shape)
        first_merge_second_branch_data = fenzu_trn_data[1].clone()
        # first_merge_third_branch_data = fenzu_trn_data[2].clone()
        # first_merge_fourth_branch_data = fenzu_trn_data[3].clone()
        first_branch = self.first_parallel(first_merge_first_branch_data, 0)
        second_branch = self.first_parallel(first_merge_second_branch_data, 1)
        # third_branch = self.first_parallel(first_merge_third_branch_data, 2)
        # fourth_branch = self.first_parallel(first_merge_fourth_branch_data, 3)

        first_merge_1 = first_branch + second_branch
        # first_merge_2 = third_branch + fourth_branch
        # second_merge = self.second_parallel(first_merge_1, 0) + self.second_parallel(first_merge_2, 1)
        if self.fenlei:
            output = self.classifier(first_merge_1)
            return nn.functional.log_softmax(output)
        else:
            return first_merge_1

    def first_parallel(self, input_to_give, index):
        """
        :param input_to_give:
        :param index: 融合过程，第几个分支（不同分支参数不共享，结构相同但都要重新定义）
        :return:
        """
        return self.first_part_FC[index](input_to_give)

    def second_parallel(self, input_to_give, index):
        """
        :param input_to_give:
        :param index: 融合过程，第几个分支（不同分支参数不共享，结构相同但都要重新定义）
        :return:
        """
        return self.second_part_FC[index](input_to_give)



