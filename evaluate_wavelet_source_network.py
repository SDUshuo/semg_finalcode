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

import LMF
import TCN as TCNmodule
import myTRN
import TRNmodule
from save_data import number_of_vector_per_example,number_of_classes,size_non_overlap,number_of_canals

batchsize=512
"""
该程序是一个基于Pytorch实现的使用小波CNN对数据集进行分类的程序。
该程序实现了扰动数据的训练、测试、计算混淆矩阵、计算模式以及权重初始化等任务。程序通过对多个数据集的训练和测试来计算模型的整体准确性。
"""
def confusion_matrix(pred, Y, number_class=7):
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0)
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)


def scramble(examples, labels, second_labels=[]):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def calculate_fitness(examples_training, labels_training,examples_test_0, labels_test_0,examples_test_1,
                      labels_test_1,examples_training_TCN, examples_validation0_TCN,examples_validation1_TCN):
    accuracy_test0 = [] # 存储测试集0的准确率
    accuracy_test1 = []# 存储测试集1的准确率



    """定义各类模型 """
    trainmode = 'tcn'  # else tcn
    # # 创建slowfushion
    # TRN = Wavelet_CNN_Source_Network.Net(number_of_class=7, batch_size=batchsize, number_of_channel=12,
    #                                      learning_rate=0.0404709, dropout=.5).cuda()
    # 创建TCN
    TRN = TCNmodule.TemporalConvNet(num_inputs=8, num_channels=[40,15, 8, 1], fc1_dim=number_of_vector_per_example,
                                    fc2_dim=number_of_vector_per_example, class_num=number_of_classes,
                                    kernel_size=2, dropout=0.3).cuda()
    seed = 3
    # #创建LMF多模态融合
    # fushion_dim = 128
    # rank = 4
    # hidden_dim = 64
    # class_nums = 7
    # LMF = model.LMF.LMF((100, 52), fushion_dim, rank, hidden_dim, class_nums).cuda()
    # torch.autograd.set_detect_anomaly(True)
    # 创建 TRN模型
    # TRN= myTRN.Net(number_of_class=7,num_segments=12, batch_size=batchsize,
    #                                      learning_rate=0.0404709, dropout=.5).cuda()

    for dataset_index in range(1, 18):
        # 在每次循环开始之前设置随机数种子
        setup_seed(seed)
        # 准备用于微调的训练集数据
        #下面三个for循环是为了保证在所有的数据都加载，最外层for dataset_index in range(0, 18):是最多的数据集的长度
        # 当测试集加载完毕后test0和test1两个for就没用了，不用管
        X_fine_tune_train, Y_fine_tune_train, X_TCN_fine_tune_train = [], [], []
        for label_index in range(len(labels_training)):
            if label_index == dataset_index:#label_index最外层索引是人类编号
                print("Current dataset test : ", dataset_index)
                for example_index in range(len(examples_training[label_index])):#这里指的是28个dat编号
                    if (example_index < 28):
                        X_fine_tune_train.extend(examples_training[label_index][example_index])
                        Y_fine_tune_train.extend(labels_training[label_index][example_index])
                        X_TCN_fine_tune_train.extend(examples_training_TCN[label_index][example_index])

        X_test_0, Y_test_0, X_TCN_fine_tune_test = [], [], []
        # 准备测试集0的数据
        for label_index in range(len(labels_test_0)):
            if label_index == dataset_index:
                for example_index in range(len(examples_test_0[label_index])):
                    X_test_0.extend(examples_test_0[label_index][example_index])
                    Y_test_0.extend(labels_test_0[label_index][example_index])
                    X_TCN_fine_tune_test.extend(examples_validation0_TCN[label_index][example_index])
        # 打乱用于微调的训练数据
        X_fine_tune, Y_fine_tune,X_fine_tune_TCN = scramble(X_fine_tune_train, Y_fine_tune_train,X_TCN_fine_tune_train)

        val_scale=0.1
        # 划分验证集
        valid_examples = X_fine_tune[0:int(len(X_fine_tune) * val_scale)]
        labels_valid = Y_fine_tune[0:int(len(Y_fine_tune) * val_scale)]
        valid_examples_TCN=X_fine_tune_TCN[0:int(len(X_fine_tune_TCN) * val_scale)]

        X_fine_tune = X_fine_tune[int(len(X_fine_tune) * val_scale):]
        Y_fine_tune = Y_fine_tune[int(len(Y_fine_tune) * val_scale):]
        X_fine_tune_TCN=X_fine_tune_TCN[int(len(X_fine_tune_TCN) * val_scale):]


        # 转换为TensorDataset对象
        print(torch.from_numpy(np.array(Y_fine_tune, dtype=np.int32)).size(0))
        #
        print(np.shape(np.array(X_fine_tune, dtype=np.float32)))
        train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_fine_tune, dtype=np.int32)),
                              torch.from_numpy(np.array(X_fine_tune_TCN, dtype=np.float32)))
        validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_valid, dtype=np.int32)),
                                   torch.from_numpy(np.array(valid_examples_TCN, dtype=np.float32)))
        # 创建数据加载器
        trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=batchsize, shuffle=True)

        # 创建测试集的数据加载器
        test_0 = TensorDataset(torch.from_numpy(np.array(X_test_0, dtype=np.float32)),
                               torch.from_numpy(np.array(Y_test_0, dtype=np.int32)),
                               torch.from_numpy(np.array(X_TCN_fine_tune_test, dtype=np.float32)),)


        test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=1, shuffle=False)



        """
        定义损失函数
        """
        criterion = nn.NLLLoss(size_average=False)
        precision = 1e-8


        # optimizer = optim.Adam(list(cnn.parameters()) + list(LMF.parameters())+list(TCN.parameters()), lr=0.0404709)
        optimizer = optim.Adam(TRN.parameters(), lr=0.008)
        #学习率调度器（scheduler）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)


        """
        进行训练
        """
        cnn=[]
        TCN=[]
        LMF =[]
        models=[cnn,TCN,LMF,TRN]

        want_operation='train'

        if want_operation=='train':
            multi_model = train_model(models,criterion, optimizer, scheduler,trainmode,
                              dataloaders={"train": trainloader, "val": validationloader}, precision=precision)
        else:
            cnn_weights = torch.load('best_weights_source_wavelet.pt')
            TRN.load_state_dict(cnn_weights)
            multi_model=TRN
        # print(multi_model)
        multi_model.eval()
        total = 0
        correct_prediction_test_0 = 0


        """
        进行测试集测试
        """
        for k, data_test_0 in enumerate(test_0_loader, 0):
            # get the inputs
            inputs_test_0, ground_truth_test_0,inputs_test_0_TCN = data_test_0
            inputs_test_0, ground_truth_test_0,inputs_test_0_TCN = Variable(inputs_test_0.cuda()), Variable(ground_truth_test_0.cuda()),Variable(inputs_test_0_TCN.cuda())

            #对trn测试
            # concat_input = inputs_test_0
            #对TCN测试
            if trainmode == 'tcn':
                concat_input = inputs_test_0_TCN
                for i in range(20):
                    concat_input = torch.cat([concat_input, inputs_test_0_TCN])
            else:
                concat_input = inputs_test_0
                for i in range(20):
                    concat_input = torch.cat([concat_input, inputs_test_0_TCN])
            #将 inputs_test_0 连接（concatenate） 20 次，形成 concat_input，目的可能是扩充输入数据量，把test的数据一次20batchsize送进去。


            outputs_test_0 = multi_model(concat_input)
            _, predicted = torch.max(outputs_test_0.data, 1)
            #将预测结果和真实标签进行比较，计算正确预测的数量 correct_prediction_test_0。
            # 这里使用了 mode() 函数来获取预测结果中出现最多的元素，并与真实标签进行比较。
            correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
                                          ground_truth_test_0.data.cpu().numpy()).sum()
            total += ground_truth_test_0.size(0)# 总样本数量
        print("ACCURACY TEST_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))
        accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))



    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())

    return accuracy_test0


def train_model(models,criterion, optimizer, scheduler, trainmode,dataloaders, num_epochs=100, precision=1e-8):
    since = time.time()
    cnn = models[0]
    TRN=models[3]
    best_loss = float('inf')

    #耐心值 控制早停 。
    # 当验证损失值连续若干个周期没有改善时，耐心值逐渐减少，如果达到了设定的耐心值阈值，就会停止训练。
    patience = 30
    patience_increase = 10
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 对于每个epoch都有train和val环节
        for phase in ['train', 'val']:
            if phase == 'train':
                TRN.train(True)
            else:
                TRN.eval()
                # cnn.train(False)  # Set model to evaluate mode
                # TCN.train(False)
                # LMF.train(False)
            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels,TCN_input = data

                inputs, labels ,TCN_input= Variable(inputs.cuda()), Variable(labels.cuda().long()),Variable(TCN_input.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':

                    TRN.train(True)
                    """
                    进行模型的训练
                    """
                    #slowfushion 特征生成
                    if trainmode=='tcn':
                        img_outputs = TRN(TCN_input)
                    else :
                        img_outputs= TRN(inputs)
                    #获取两个模态的维度
                    # outputs= LMF(img_outputs,TCN_output)

                    # _, predictions = torch.max(outputs.data, 1)
                    # loss = criterion(outputs, labels)
                    _, predictions = torch.max(img_outputs.data, 1)
                    loss = criterion(img_outputs, labels)

                    loss.backward()

                    optimizer.step()
                    loss = loss.item()

                else:
                    TRN.eval()

                    accumulated_predicted = Variable(torch.zeros(len(inputs), 7)).cuda()
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(20):
                        # slowfushion 特征生成
                        if trainmode == 'tcn':
                            img_outputs = TRN(TCN_input)
                        else:
                            img_outputs = TRN(inputs)
                        # TCN_output = TCN(TCN_input)
                        # 获取两个模态的维度
                        # outputs = LMF(img_outputs, TCN_output)

                        loss = criterion(img_outputs, labels) #img_outputs(batch_size, num_classes) labels(batch_size,)，

                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(img_outputs.data, 1)#返回最大值和对应的索引。这里只是用到了索引，即给他预测了第几类
                        #根据每次子模型的预测结果，在accumulated_predicted中对应位置加1，实现预测结果的累加。
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary/total_sub_pass



                # statistics
                running_loss += loss #累积每个batch的损失值。
                running_corrects += torch.sum(predictions == labels.data) #统计预测正确的样本数量
                total += labels.size(0) #累积样本的总数。

            epoch_loss = running_loss / total #当前epoch的平均损失值。
            epoch_acc = running_corrects / total  #当前epoch的准确率。
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)#根据当前epoch的验证损失值来调整学习率，使用预定义的学习率调整策略。
                if epoch_loss+precision < best_loss:  #当前验证损失值加上一个小的精度值小于最佳损失值best_loss
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    #在验证集上获得最佳损失时保存模型，并将其保存为文件
                    torch.save( TRN.state_dict(), 'best_weights_source_wavelet.pt')
                    patience = patience_increase + epoch #更新耐心值为当前epoch加上预定义的耐心增加值。
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience: #如果当前epoch大于耐心值，则跳出循环，结束训练过程
            break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
   