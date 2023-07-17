"""18个类别（53个类别也是这个)   83.071 % """
# # LMF多模态融合 参数
# fushion_dim = 80 #两个特征融合后的维度
# rank = 6
# hidden_dim = 64 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 80  #多尺度TRN源码输出的维度
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度
#
# #TCN参数
# num_channels=[64, 64, 32, 18,1]

""" 79.311 %"""
# # LMF多模态融合 参数
# fushion_dim = 100 #两个特征融合后的维度（fushion_2_feature_bottleneck_，number_of_class=18）
# rank = 6
# hidden_dim = 80 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 80  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度
#
# #TCN参数
# num_channels=[64, 64, 32, 18,1]
"""75.673 %"""
# #这个层数是往低了调，效果反而不好
# # LMF多模态融合 参数
# fushion_dim = 64 #两个特征融合后的维度（fushion_2_feature_bottleneck_，number_of_class=18）
# rank = 4
# hidden_dim = 30 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 100  #多尺度TRN源码输出的维度 40维度的图片传入TRN =>> nn.Linear(4 * self.img_feature_dim (160),  feature_bottleneck),
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度
#
# #TCN参数
# num_channels=[64, 64, 32, 18,1]
"""改变TCN效果变好不少 前两个可以少到50，一个列表内不能都是一样的，比如前四个都是五十  84.068 %"""
# # LMF多模态融合 参数
# fushion_dim = 64 #两个特征融合后的维度
# rank = 6
# hidden_dim = 64 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 64  #多尺度TRN源码输出的维度
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度
#
# #TCN参数
# num_channels=[64, 64, 32, 32,1]
""" 84.662 % """
# # LMF多模态融合 参数
# fushion_dim = 64 #两个特征融合后的维度
# rank = 6
# hidden_dim = 64 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 64  #多尺度TRN源码输出的维度
# fushion_2_feature_bottleneck_=64  #myTRN输出的维度
#
# #TCN参数
# num_channels=[50, 50, 32, 32,1]
""" 此时的融合还是+而不是contact 84.750 %"""
# # LMF多模态融合 参数
# fushion_dim = 64 #两个特征融合后的维度
# rank = 6
# hidden_dim = 64 #全连接层分类，前一层的维度
#
# # TRN 参数
# feature_bottleneck = 64  #多尺度TRN源码输出的维度
# fushion_2_feature_bottleneck_=64  #这个*2就是myTRN输出的维度 128
#
# #TCN参数
# num_channels=[50, 50, 28, 28,1]  #TCN输出的一直都是52

