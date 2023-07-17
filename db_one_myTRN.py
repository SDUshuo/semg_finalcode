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
from params_contact import fushion_2_feature_bottleneck_,feature_bottleneck
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
        self.trn = TRNmodule.return_TRN('TRNmultiscale', self.img_feature_dim, 4, number_of_class, self.feature_bottleneck)

        self.fushion_2_feature_bottleneck_=fushion_2_feature_bottleneck_ #mytrn第二次融合的向量长度
        self.first_part_FC=[]
        for i in range(3):
            self.first_part_FC.append(nn.Sequential(
                nn.Linear(self.feature_bottleneck, self.feature_bottleneck),
                # nn.BatchNorm1d(self.feature_bottleneck),
                nn.ReLU(),
                nn.Dropout()
            ).cuda())
        self.second_part_FC = []

        self.second_part_FC.append(nn.Sequential(
            nn.Linear(self.feature_bottleneck, self.fushion_2_feature_bottleneck_),
            # nn.BatchNorm1d(self.fushion_2_feature_bottleneck_),
            nn.ReLU(),
            nn.Dropout()
        ).cuda())
        self.second_part_FC.append(nn.Sequential(
            nn.Linear(self.feature_bottleneck, self.fushion_2_feature_bottleneck_),
            # nn.BatchNorm1d(self.fushion_2_feature_bottleneck_),
            nn.ReLU(),
            nn.Dropout()
        ).cuda())
        self.num_class=number_of_class
        self.classifier = nn.Sequential(
            nn.Linear(self.fushion_2_feature_bottleneck_, self.fushion_2_feature_bottleneck_),
            nn.BatchNorm1d(self.fushion_2_feature_bottleneck_),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fushion_2_feature_bottleneck_, self.num_class)
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

        #12帧图片
        for i in range(self.num_segments):
            #[(batchsize,feature_dim),]
            # x[:, i, :, :].unsqueeze(1).shape: torch.Size([batchsize, 1, 8, 7])
            '''去掉[i]就行'''
            conv_feature=self.all_first_conv(x[:, i, :, :].unsqueeze(1)).cuda().view(x.shape[0], -1)#torch.Size([batchsize, 40])  [512, 216]
            # conv_feature = self.all_first_conv(x[:, i, :, :].unsqueeze(1)).cuda().view(x.shape[0],
            #                                                                               -1)  # torch.Size([batchsize, 40])
            # print(conv_feature.shape)
            all_x_flattend.append(conv_feature)

        #将12帧分为三组，每组四帧
        all_fenzu_data = [all_x_flattend[i:i + 4] for i in range(0, len(all_x_flattend), 4)]
        fenzu_trn_data=[]
        #每组四帧图片送入trn中，输入维度[batchsize,4,feature_dim]  如果12帧，那就是三组
        for fenzu_data in all_fenzu_data:
            #fenzu_data 列表 4个tensor
            combined_tensor = torch.stack(fenzu_data, dim=0).transpose(0, 1) #torch.Size([batchsize, 4, 40])

            fenzu_trn_data.append(self.trn( combined_tensor))
        # for i in range(len(fenzu_trn_data)):
        #     fenzu_trn_data[i] = fenzu_trn_data[i].cuda()
        """
        进行慢融合阶段
        """
        #第一次融合
        # print(fenzu_trn_data[0].shape)
        first_merge_first_branch_data=fenzu_trn_data[0].clone()
        # print(first_merge_first_branch_data.shape)
        first_merge_second_branch_data=fenzu_trn_data[1].clone()
        first_merge_third_branch_data = fenzu_trn_data[2].clone()

        first_branch = self.first_parallel(first_merge_first_branch_data, 0)
        second_branch = self.first_parallel(first_merge_second_branch_data, 1)
        third_branch = self.first_parallel(first_merge_third_branch_data, 2)
        first_merge_1 = first_branch + second_branch
        first_merge_2 = second_branch + third_branch

        # first_merge_1 = torch.cat([first_branch,second_branch],dim=-1)
        # first_merge_2 = torch.cat([second_branch, third_branch], dim=-1)
        #第二次融合
        # second_merge_data=fenzu_trn_data[2].clone()
        second_merge =self.second_parallel(first_merge_1, 0) + self.second_parallel(first_merge_2, 1)
        # second_merge = torch.cat([self.second_parallel(first_merge_1, 0),self.second_parallel(first_merge_2, 1)],dim=-1)
        # second_merge = torch.cat([self.second_parallel(first_merge_1, 0),self.second_parallel(second_merge_data, 1)],dim=-1)
        if self.fenlei:
            output = self.classifier(second_merge)
            return nn.functional.log_softmax(output)
        else:
            return second_merge

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



