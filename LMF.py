import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dim_tuple, fushion_dim, rank,hidden_dim, class_nums,use_softmax=True):
        '''
        Args:

            input_dim_tuple - another length-2 tuple, 多模态输入的两个特征向量的维度
            dropouts - a length-3 tuple, contains (image_dropout, time_dropout, post_fusion_dropout)
            fushion_dim - int, specifying the size of output  输出的融合特征向量维度
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of image, time and text

        self.image_hidden = input_dim_tuple[0]
        self.time_hidden = input_dim_tuple[1]
        self.fushion_dim = fushion_dim
        #R
        self.rank = rank
        self.use_softmax = use_softmax

        self.image_factor = Parameter(torch.Tensor(self.rank, self.image_hidden+1, self.fushion_dim)) # Wa = Parameter(torch.Tensor(R, A.shape[1], h))
        self.time_factor = Parameter(torch.Tensor(self.rank, self.time_hidden+1, self.fushion_dim))

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))  #Wf = Parameter(torch.Tensor(1, R))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fushion_dim))#bias = Parameter(torch.Tensor(1, h))

        # init teh factors
        xavier_normal(self.image_factor)
        xavier_normal(self.time_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        #全连接层分类
        self.fc1 = nn.Linear(fushion_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, class_nums)
    def forward(self, image_x, time_x):
        '''
        Args:
            image_x: tensor of shape (batch_size, image_in)
            time_x: tensor of shape (batch_size, time_in)
        '''
        image_h =image_x
        time_h = time_x
        batch_size = image_h.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if image_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        # _image_h = torch.cat([image_h,(Variable(torch.ones(batch_size, 1).type(DTYPE),requires_grad=False))], dim=1)
        # _time_h = torch.cat([time_h,(Variable(torch.ones(batch_size, 1).type(DTYPE),requires_grad=False))], dim=1)
        _image_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image_h), dim=1)
        _time_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), time_h), dim=1)

        # 分解后，并行提取各模态特征
        # print(image_h.shape)
        # print(_image_h.shape)
        # print(self.image_factor.shape)
        fusion_image = torch.matmul(_image_h, self.image_factor)#fusion_A = torch.matmul(A, Wa)
        fusion_time = torch.matmul(_time_h, self.time_factor)
        # 利用一个Linear再进行特征融合（融合R维度）
        fusion_zy = fusion_image * fusion_time

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.fushion_dim)
        #全连接分类
        out = self.fc2(self.dropout(self.relu(self.bn1(self.fc1(output)))))
        if self.use_softmax:
            out = nn.functional.log_softmax(out, dim=1)

        # print(out.shape)
        # torch.Size([512, 53])
        return out

