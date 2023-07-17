# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck,self.num_class),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]
    """
    :param num_frames 帧总数 （例如：12）
    :param img_feature_dim 图片向量嵌入 维度
    :param num_class 分类
    """
    def __init__(self, img_feature_dim, num_frames, num_class,num_bottleneck):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim

        #如果num_frames为12，则根据代码 self.scales = [i for i in range(num_frames, 1, -1)]，
        # 生成的self.scales列表为[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]。这表示使用12帧关系、11帧关系、10帧关系、9帧关系、8帧关系、7帧关系、6帧关系、5帧关系、4帧关系、3帧关系和2帧关系。
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            #对所有帧图片，按照尺度，生成全排列
            relations_scale = self.return_relationset(num_frames, scale) #[(),()]
            self.relations_scales.append(relations_scale)
            #根据全排列数，最大是三，也就是说，对于一个尺度来说，最多取三个关系
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        self.num_bottleneck = num_bottleneck #隐藏层维度

        #对于每个尺度，创建一个全连接神经网络模型fc_fusion，并将其添加到self.fc_fusion_scales中。
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim,  self.num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        nn.Linear( self.num_bottleneck,  self.num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6)
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])
        # print(self)
        #
        # print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params
    #input:(batch_size, num_frames, img_feature_dim)。
    def forward(self, input):
        # the first one is the largest scale
        #从输入中选择最大尺度的关系集合中的第一个关系，并将其存储在act_all中
        act_all = input[:, self.relations_scales[0][0] , :].clone()
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        #将act_all展平后传入对应的全连接神经网络模型self.fc_fusion_scales[0]进行预测。
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                          self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all = torch.add(act_all, act_relation)

        return act_all

    #返回一个关系集合的列表，其中包含所有从num_frames个帧中选择num_frames_relation个帧的组合。
    #生成全排列（按照顺序）num_frames=4和num_frames_relation=2 时[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class RelationModuleMultiScaleWithClassifier(torch.nn.Module):
    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScaleWithClassifier, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

def return_TRN(relation_type, img_feature_dim, num_frames, num_class,feature_bottleneck):
    if relation_type == 'TRN':
        TRNmodel = RelationModule(img_feature_dim, num_frames, num_class)
    elif relation_type == 'TRNmultiscale':
        TRNmodel = RelationModuleMultiScale(img_feature_dim, num_frames, num_class,feature_bottleneck)
    else:
        raise ValueError('Unknown TRN' + relation_type)


    return TRNmodel

if __name__ == "__main__":
    batch_size = 10
    num_frames = 5
    num_class = 174
    img_feature_dim = 512
    input_var = Variable(torch.randn(batch_size, num_frames, img_feature_dim))
    model = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(output)