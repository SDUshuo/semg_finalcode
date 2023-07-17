import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from McDropout import McDropout
import numpy as np
# import home.yanshuo.YS.FinalCode.PyTorchImplementation.CWT.model.TRNmodule as TRNmodule

# import PyTorchImplementation.CWT.model.TRNmodule as TRNmodule
import TRNmodule
# from TRNmodule import return_TRN

class Net(nn.Module):
    def __init__(self, number_of_class,num_segments=12, fenlei=True):
        super(Net, self).__init__()
        self.fenlei=fenlei
        self._input_batch_norm = nn.BatchNorm2d(num_segments, eps=1e-4)
        #self._input_prelu = pelu((1, 12, 1, 1))
        self._input_prelu = nn.PReLU(num_segments)

        self.all_first_conv=nn.Conv2d(1, 1, kernel_size=3)
        self.img_feature_dim=30
        self.num_segments=num_segments #有多少帧
        self.feature_bottleneck = 64
        self.trn = TRNmodule.return_TRN('TRNmultiscale', self.img_feature_dim, 4, number_of_class, self.feature_bottleneck)

        self.fushion_2_feature_bottleneck_=64 #mytrn第二次融合的向量长度
        self.first_part_FC=[]
        for i in range(2):
            self.first_part_FC.append(nn.Sequential(
                            nn.Linear(self.feature_bottleneck, self.feature_bottleneck),
                            nn.ReLU(),
                            nn.Dropout()
                            ).cuda())
        self.second_part_FC = []
        for i in range(2):
            self.second_part_FC.append(nn.Sequential(
                nn.Linear(self.feature_bottleneck, self.fushion_2_feature_bottleneck_),
                nn.ReLU(),
                nn.Dropout()
            ).cuda())
        self.num_class=number_of_class
        self.classifier = nn.Linear(self.fushion_2_feature_bottleneck_, self.num_class)
        # print(self)
        #
        # print("Number Parameters: ", self.get_n_params())
    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

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
        x：[batchsize,12,8,7]
        对数据处理，对每个批次包含的12帧的图片化成12个向量，送入trn，返回[batchsize,dim]
        """

        x = self._input_prelu(self._input_batch_norm(x))
        all_x_flattend=[]

        #12帧图片
        for i in range(self.num_segments):
            #[(batchsize,feature_dim),]
            # x[:, i, :, :].unsqueeze(1).shape: torch.Size([128, 1, 8, 7])
            conv_feature=self.all_first_conv(x[:, i, :, :].unsqueeze(1)).cuda().view(-1, 30)#torch.Size([128, 30])
            all_x_flattend.append(conv_feature)

        #将12帧分为三组，每组四帧
        all_fenzu_data = [all_x_flattend[i:i + 4] for i in range(0, len(all_x_flattend), 4)]
        fenzu_trn_data=[]
        #每组四帧图片送入trn中，输入维度[batchsize,4,feature_dim]  如果12帧，那就是三组
        for fenzu_data in all_fenzu_data:
            #fenzu_data 列表 4个tensor
            combined_tensor = torch.stack(fenzu_data, dim=0).transpose(0, 1) #torch.Size([128, 4, 30])
            fenzu_trn_data.append(self.trn( combined_tensor))
        # for i in range(len(fenzu_trn_data)):
        #     fenzu_trn_data[i] = fenzu_trn_data[i].cuda()
        """
        进行慢融合阶段
        """
        #第一次融合
        # print(fenzu_trn_data[0].shape)
        first_merge_first_branch_data=fenzu_trn_data[0].clone()
        first_merge_second_branch_data=fenzu_trn_data[1].clone()

        first_branch = self.first_parallel(first_merge_first_branch_data, 0)
        second_branch = self.first_parallel(first_merge_second_branch_data, 1)
        first_merge_1 = first_branch + second_branch
        #第二次融合
        second_merge_data=fenzu_trn_data[2].clone()
        second_merge = self.second_parallel(first_merge_1, 0) + self.second_parallel(second_merge_data, 1)
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



