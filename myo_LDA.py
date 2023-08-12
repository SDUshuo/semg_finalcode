import torch
import torch.nn as nn
import torch.optim as optim

import random
import nolds
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.extmath import softmax
import Wavelet_CNN_Source_Network as Wavelet_CNN_Source_Network
from torch.utils.data import TensorDataset

import torch
from torch.autograd import Variable
import time
from torch.nn import functional as F, TransformerEncoderLayer, TransformerEncoder
from scipy.stats import mode
import db_one_model as db_one_model
import LMF as LMFmodule
from sklearn import svm
import TCN as TCNmodule
import myTRN as myTRN
import TRNmodule
import params_contact
from Ninapro_LDA import extract_TD_features, top_n_accuracy, extract_features
from save_data import number_of_vector_per_example, number_of_classes, size_non_overlap, number_of_canals
import os
from load_evaluation_dataset_DB1 import newpath, window_inc, window_len, window_path

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
    accuracy_test0 = []  # 存储测试集0的准确率
    top3_accuracy_test0 = []

    seed = 3

    # #创建slowfushion
    # CNN = Wavelet_CNN_Source_Network.Net(number_of_class=7, batch_size=batchsize, number_of_channel=12,
    #                                      learning_rate=0.0404709, dropout=.5).cuda()
    setup_seed(seed)


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


        test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=batchsize, shuffle=False)
        X_train =extract_features(np.array(X_fine_tune_TCN, dtype=np.float32)).squeeze().transpose()
        Y_train =np.array(Y_fine_tune, dtype=np.int32)
        X_test=extract_features(np.array(X_TCN_fine_tune_test, dtype=np.float32)).squeeze().transpose()
        Y_test= np.array(Y_test_0, dtype=np.int32)
        # TD_features = extract_TD_features(np.array(X_fine_tune_TCN, dtype=np.float32)).squeeze()
        # print(TD_features.shape)
        # X_train = TD_features.reshape(len(TD_features), -1)
        # print(X_train.shape)
        # Y_train = np.array(Y_fine_tune, dtype=np.int32)
        # TD_features_test = extract_TD_features(np.array(X_TCN_fine_tune_test, dtype=np.float32)).squeeze()
        # X_test = TD_features_test.reshape(len(TD_features_test), -1)
        # Y_test = np.array(Y_test_0, dtype=np.int32)
        clf = LinearDiscriminantAnalysis()
        print(np.array(Y_fine_tune, dtype=np.int32).shape)
        clf.fit(X_train, Y_train)
        accuracy = clf.score(X_test, Y_test)
        print('Test accuracy: ', accuracy)
        accuracy_test0.append(accuracy)
        # Make predictions
        y_scores = clf.decision_function(X_test)
        y_proba = softmax(y_scores)

        # Calculate top-3 accuracy
        top_3_accuracy = top_n_accuracy(y_proba, Y_test, 3)
        top3_accuracy_test0.append(top_3_accuracy)
    return accuracy_test0,top3_accuracy_test0

#models=[CNN,TCN,LMF,TRN]
def train_model(model,criterion, optimizer, scheduler,dataloaders, num_epochs=150, precision=1e-8):
    since = time.time()
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
                model.train(True)
            else:
                model.eval()

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
                    """
                    进行模型的训练
                    """
                    outputs = model(inputs, TCN_input)

                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)


                    loss.backward()

                    optimizer.step()
                    loss = loss.item()

                else:
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 7)).cuda()
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(20):

                        outputs = model(inputs, TCN_input)
                        _, predictions = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)

                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)#返回最大值和对应的索引。这里只是用到了索引，即给他预测了第几类
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
                    torch.save( model.state_dict(), 'best_weights_source_wavelet.pt')
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
    # load best model weights
    model_weights = torch.load('best_weights_source_wavelet.pt')
    model.load_state_dict(model_weights)
    model.eval()
    return  model


if __name__ == '__main__':


    # Comment between here
    #     accuracy_test_0, accuracy_test_1 = calculate_fitness(examples_training, labels_training,
    #                                                            examples_validation0, labels_validation0,
    #                                                            examples_validation1, labels_validation1)
    # examples_training,labels_training=np.load("saved_pre_training_dataset.npy", allow_pickle=True)
    # examples_validation0, labels_validation0=pickle.load(open("saved_dataset_test0.p", "rb"))
    # examples_validation1, labels_validation1 = pickle.load(open("saved_dataset_test1.p", "rb"))
    # # calculate_fitness(examples_training, labels_training,
    #                                                            examples_validation0, labels_validation0,
    #                                                            examples_validation1, labels_validation1)
    # and here if the evaluation dataset was already processed and saved with "load_evaluation_dataset"

#YS/FinalCode/PyTorchImplementation/CWT
    # print(os.listdir("../"))
    rootpath="saved_data/52_5_shift_electrodes/"
    train_datapath=rootpath+"saved_training_dataset.npy"
    test0_datapath=rootpath+"saved_Test0_dataset.npy"
    test1_datapath = rootpath+"saved_Test1_dataset.npy"

    datasets_training = np.load(train_datapath, encoding="bytes",allow_pickle=True)
    examples_training,labels_training,TCN_examples_training=np.load(train_datapath, encoding="bytes",allow_pickle=True)
    examples_validation0, labels_validation0,TCN_examples_validation0=np.load(test0_datapath, encoding="bytes",allow_pickle=True)
    examples_validation1, labels_validation1,TCN_examples_validation1 = np.load(test1_datapath, encoding="bytes",allow_pickle=True)
    print("SHAPE of training:   ", np.shape(examples_training))
    print("++++++++++++")
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []

    test_0 = []
    top3_test_0 = []
    test_1 = []

    for i in range(1):
        accuracy_test_0,top3_accuracy_test0  = calculate_fitness(examples_training, labels_training,
                                                               examples_validation0, labels_validation0,examples_validation1, labels_validation1,
                                                               TCN_examples_training, TCN_examples_validation0,TCN_examples_validation1)
        print(top3_accuracy_test0)
        test_0.append(accuracy_test_0)
        top3_test_0.append(top3_accuracy_test0)

        test_0.append(accuracy_test_0)

        print("ACCURACY FINAL TEST 0: ", test_0)
        print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
        print("ACCURACY FINAL top3  TEST 0: ", np.mean(top3_test_0))

    with open("Pytorch_results_4_cycles.txt", "a") as myfile:
        myfile.write("CNN STFT: \n\n")
        myfile.write("Test 0: \n")
        myfile.write(str(np.mean(test_0, axis=0)) + '\n')
        myfile.write(str(np.mean(test_0)) + '\n')

        myfile.write("Test 1: \n")
        myfile.write(str(np.mean(test_1, axis=0)) + '\n')
        myfile.write(str(np.mean(test_1)) + '\n')
        myfile.write("\n\n\n")
