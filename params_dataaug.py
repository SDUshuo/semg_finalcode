"""  82  数据增强后提高84.593 %"""
# # LMF多模态融合 参数
# fushion_dim = 64 #两个特征融合后的维度 （fushion_2_feature_bottleneck_，number_of_class=18）
# rank = 6
# hidden_dim = 64 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 64  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=64  #这个*2就是myTRN输出的维度 128
#
# #TCN参数
# num_channels=[50, 50, 28, 28,1]  #TCN输出的一直都是52
"""改回+ NCE 82"""
# # LMF多模态融合 参数
# fushion_dim = 64  #两个特征融合后的维度 （fushion_2_feature_bottleneck_,52）
# rank = 6
# hidden_dim = 64 #全连接层分类，前一层的维度
# NCE_feature_dim = 80  # 80  82.127 %
# # TRN 参数
# feature_bottleneck = 100  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度
# #TCN参数
# num_channels=[50, 50, 32, 32,1]  #TCN输出的一直都是52

"""接下来就是单独对TCN进行解耦实验"""
# num_channels=[50, 50, 32, 32,1]  84%  这个已经很好了
"""对TRN进行解耦实验"""
# #TCN参数
num_channels=[50, 50, 30,1]  #TCN输出的一直都是52
# TRN 参数
# #100-77%
# feature_bottleneck = 100  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度 这个时候fushion_2_feature_bottleneck_没考虑，self.first_part_FC.append(linear(self.feature_bottleneck, self.feature_bottleneck),
#feature_bottleneck和fushion_2_feature_bottleneck_应该相等比较好
#120 -77%
#90 - 75%3
# #  尝试 cat   100-77.562 %
# feature_bottleneck = 100  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=100  #myTRN输出的维度
'''尝试 attention（这个最好，参数少，效果好）    100-81.165 %  110-81.410 % 140-81.812 %  160-83.106 %（参数298k）'''
attention_dim=16    #在feature_bottleneck = 160的前提下：16-83%   30-82%  20-82.267 %  10-83.088 %
# feature_bottleneck = 180  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
fushion_2_feature_bottleneck_=100  #myTRN输出的维度
'''attention  修改四个张量为contact  160-82.459 %  80-82.756 %  最后又改成sum了'''
# attention_dim=16    #在feature_bottleneck = 160的前提下：16-83%   30-82%  20-82.267 %  10-83.088 %
# feature_bottleneck = 160  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=160  #myTRN输出的维度
'''attention  修改为四尺度三张量attention   160-82%（参数340k）  180- 82.5%  '''
# attention_dim=16    #在feature_bottleneck = 160的前提下：16-83%   30-82%  20-82.267 %  10-83.088 %
# feature_bottleneck = 180  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=180  #myTRN输出的维度
'''attention  修改为6尺度2张量attention   参数量太大  '''
# attention_dim=16    #在feature_bottleneck = 160的前提下：16-83%   30-82%  20-82.267 %  10-83.088 %
# feature_bottleneck = 180  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=180  #myTRN输出的维度

'''尝试 early，对12个先 attention    三尺度都一般 6也一般，舍弃'''
# attention_dim=16    #在feature_bottleneck = 160的前提下：16-83%   30-82%  20-82.267 %  10-83.088 %
# feature_bottleneck = 160  #多尺度TRN源码输出的维度  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=100  #myTRN输出的维度


"""LMF 测试  [fushion_dim,hidden_dim]： [100,100]-85.030 % [100,60]-81.567 %   [120,100]-85.222 %  [150,100]80.728 %    [140,140]-85.100 % [110,110]-80.325 %
    固定为[100,100]  对attention的feature_bottleneck进行更改  上面的这些都是在160下实现的， 100-81.182 %   140-79.941 %  180-86.009 %  200-85.817 % 190-85.4%  最后使用180
"""
# LMF多模态融合 参数
fushion_dim = 100  #两个特征融合后的维度 （fushion_2_feature_bottleneck_,52）
rank = 6  # 10- 82.441 % 15-85.240 %
hidden_dim = 100 #全连接层分类，前一层的维度
"""不适用lmf，使用contact测试 80最好  180-85%  140-85.152 % 100-86.551 % 80-86.516 %  60-80% 120-86.4% """
feature_bottleneck = 150
"""  使用autoencoder 分类层使用cat 参数会很多，他会在60左右的epoch就达到94%，可以尝试一下epoch早停
最后使用180   150-84.348 % 180-85.88 190-84.9%"""
# encoding_dim =180
"""  使用autoencoder 分类层使用sum   180-79%
"""
encoding_dim =180
""" 使用attention做最后的融合 效果很差，放弃"""

"""加入对比学习损失函数 NCE_feature_dim 80-80% 120-81.007 %   """
NCE_feature_dim =  120 # 把trn和tcn的输出都fc到这个维度，然后再进行多模态融合
alpha=0.8

