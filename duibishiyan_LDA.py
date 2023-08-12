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
from save_data import number_of_vector_per_example, number_of_classes, size_non_overlap, number_of_canals
import os
from load_evaluation_dataset_DB1 import newpath, window_inc, window_len, window_path


def calculate_sampen_for_each_channel(data):
    # List to store SampEn for each channel
    sampen_list = []

    # Calculate SampEn for each channel
    for i in range(data.shape[1]):
        sampen = nolds.sampen(data[:, i, :])
        sampen_list.append(sampen)

    # Convert to numpy array and return
    return np.array(sampen_list)

def extract_features(data):
    # data shape: (batch_size, num_channels=10, sequence_length=52)
    rms = np.sqrt(np.mean(data ** 2, axis=-1))  # RMS feature
    # Add more features here
    # ...
    features = np.array([rms])
    return features.T  # shape: (batch_size, num_features)


# nn.functional.log_softmax(out, dim=1)
number_of_class = 18
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
batchsize = 512

d_model = 52  # input feature dimension
nhead = 4  # head number in multi-head attention
num_layers = 1  # layer number in transformer encoder
num_classes = 27  # output class number
epochs = 5
lr = 0.01


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

# 提取TD特征
def extract_TD_features(data):
    features = []
    for batch_data in data:
        batch_features = []
        for channel_data in batch_data:
            mean = np.mean(channel_data)
            variance = np.var(channel_data)
            peak = np.max(channel_data)
            peak_to_peak = np.max(channel_data) - np.min(channel_data)
            # 将这些特征放入一个列表
            batch_features.append([mean, variance, peak, peak_to_peak])
        features.append(batch_features)
    return np.array(features)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds, axis=1)[:,-n:]
    ts = truths.reshape((len(truths),1))
    successes = np.sum(best_n == ts, axis=1)
    return float(np.sum(successes)) / float(len(truths))
def calculate_fitness(seedlist):
    accuracy_test0 = []  # 存储测试集0的准确率
    top3_accuracy_test0 = []
    seed = 25
    # 4-82.651  3-
    # #创建slowfushion
    # CNN = Wavelet_CNN_Source_Network.Net(number_of_class=7, batch_size=batchsize, number_of_channel=12,
    #                                      learning_rate=0.0404709, dropout=.5).cuda()
    setup_seed(seed)
    # model = ConvLSTM().cuda()
    X, Y, X_TCN = [], [], []
    X_val, Y_val, X_TCN_val = [], [], []
    X_test, Y_test, X_TCN_test = [], [], []
    for dataset_index in range(0, 27):
        setup_seed(seed)
        # rootpath = "saved_data/52_5_shift_electrodes/"
        # train_datapath = rootpath + "saved_training_dataset.npy"
        # test0_datapath = rootpath + "saved_Test0_dataset.npy"
        #
        # X_fine_tune_train, Y_fine_tune_train, X_TCN_fine_tune_train = np.load(train_datapath, encoding="bytes",
        #                                                                     allow_pickle=True)
        #
        # X_test_0, Y_test_0, X_TCN_fine_tune_test = np.load(test0_datapath, encoding="bytes",
        #                                                                              allow_pickle=True)

        # 在每次循环开始之前设置随机数种子
        first_path = 'saved_data/DB1/' + window_path + '/' + window_path + '_exercise1_jitr_norm_relax/subject_' + str(
            dataset_index)
        # 准备训练集
        # directory = 'saved_data/DB1/52_5/subject_' + str(dataset_index) + '/train'
        # directory = 'saved_data/DB1/52_5/52_5_exercise1/subject_'+ str(dataset_index) + '/train'
        directory = first_path + '/train'
        X_fine_tune_train = np.load(directory + 'X_train_CWT.npy', encoding="bytes", allow_pickle=True)
        Y_fine_tune_train = np.load(directory + 'Y_train.npy', encoding="bytes", allow_pickle=True)
        X_TCN_fine_tune_train = np.load(directory + 'X_train.npy', encoding="bytes", allow_pickle=True)
        # 准备测试集0的数据
        # directory = 'saved_data/DB1/52_5/subject_' + str(dataset_index) + '/test'aile
        # directory =  'saved_data/DB1/52_5/52_5_exercise1/subject_'+ str(dataset_index) + '/test'
        directory = first_path + '/test'
        X_test_0 = np.load(directory + 'X_test_CWT.npy', encoding="bytes", allow_pickle=True)
        Y_test_0 = np.load(directory + 'Y_test.npy', encoding="bytes", allow_pickle=True)
        X_TCN_fine_tune_test = np.load(directory + 'X_test.npy', encoding="bytes", allow_pickle=True)

        # 打乱用于微调的训练数据

        # 打乱用于微调的训练数据
        X_fine_tune, Y_fine_tune, X_fine_tune_TCN = scramble(X_fine_tune_train, Y_fine_tune_train,
                                                             X_TCN_fine_tune_train)
        # X_fine_tune, Y_fine_tune, X_fine_tune_TCN = X_fine_tune_train, Y_fine_tune_train,X_TCN_fine_tune_train

        val_scale = 0.01
        # 划分验证集
        valid_examples = X_fine_tune[0:int(len(X_fine_tune) * val_scale)]
        labels_valid = Y_fine_tune[0:int(len(Y_fine_tune) * val_scale)]
        valid_examples_TCN = X_fine_tune_TCN[0:int(len(X_fine_tune_TCN) * val_scale)]

        X_fine_tune = X_fine_tune[int(len(X_fine_tune) * val_scale):]
        Y_fine_tune = Y_fine_tune[int(len(Y_fine_tune) * val_scale):]
        X_fine_tune_TCN = X_fine_tune_TCN[int(len(X_fine_tune_TCN) * val_scale):]

        X.append(X_fine_tune)
        Y.append(Y_fine_tune)
        X_TCN.append(X_fine_tune_TCN)

        X_val.append(valid_examples)
        Y_val.append(labels_valid)
        X_TCN_val.append(valid_examples_TCN)

        X_test.append(X_test_0)
        Y_test.append(Y_test_0)
        X_TCN_test.append(X_TCN_fine_tune_test)

    X_fine_tune = np.concatenate(X, axis=0)
    Y_fine_tune = np.concatenate(Y, axis=0)
    X_fine_tune_TCN = np.concatenate(X_TCN, axis=0)

    valid_examples = np.concatenate(X_val, axis=0)
    labels_valid = np.concatenate(Y_val, axis=0)
    valid_examples_TCN = np.concatenate(X_TCN_val, axis=0)

    X_test_0 = np.concatenate(X_test, axis=0)
    Y_test_0 = np.concatenate(Y_test, axis=0)
    X_TCN_fine_tune_test = np.concatenate(X_TCN_test, axis=0)

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
                           torch.from_numpy(np.array(X_TCN_fine_tune_test, dtype=np.float32)), )

    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=batchsize, shuffle=False)


    X_train =extract_features(np.array(X_fine_tune_TCN, dtype=np.float32)).squeeze().transpose()
    Y_train =np.array(Y_fine_tune, dtype=np.int32)
    X_test=extract_features(np.array(X_TCN_fine_tune_test, dtype=np.float32)).squeeze().transpose()
    Y_test= np.array(Y_test_0, dtype=np.int32)
    # TD_features  = extract_TD_features(np.array(X_fine_tune_TCN, dtype=np.float32)).squeeze()
    # print(TD_features.shape)
    # X_train = TD_features.reshape(len(TD_features), -1)
    # print(X_train.shape)
    # Y_train = np.array(Y_fine_tune, dtype=np.int32)
    # TD_features_test = extract_TD_features(np.array(X_TCN_fine_tune_test, dtype=np.float32)).squeeze()
    # X_test=TD_features_test.reshape(len(TD_features_test), -1)
    # Y_test = np.array(Y_test_0, dtype=np.int32)

    clf = LinearDiscriminantAnalysis()
    print(np.array(Y_fine_tune, dtype=np.int32).shape)
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test,Y_test)
    print('Test accuracy: ', accuracy)
    accuracy_test0.append(accuracy)
    # Make predictions
    y_scores = clf.decision_function(X_test)
    y_proba = softmax(y_scores)

    # Calculate top-3 accuracy
    top_3_accuracy = top_n_accuracy(y_proba, Y_test, 3)
    top3_accuracy_test0.append(top_3_accuracy)
    return accuracy_test0,top3_accuracy_test0

'''
TD_features
Test accuracy:  0.715917076598735
[0.7938090241343126, 0.7809118325350364, 0.8380007089684509, 0.7476267239835214, 0.6632796075683252, 0.7054999121419786, 0.7975601131541725, 0.8145601974264057, 0.7441281138790036, 0.7621487895387877, 0.7657342657342657, 0.7655645444952515, 0.8315512708150745, 0.7358655697531945, 0.7748017621145374, 0.8196548080309969, 0.7877457865168539, 0.7508021390374332, 0.740252803987894, 0.7863636363636364, 0.7867711053089643, 0.7833690221270521, 0.7460429124164615, 0.7573185518794388, 0.7876997915218902, 0.8155579021471313, 0.715917076598735]
ACCURACY FINAL TEST 0:  [[0.7938090241343126, 0.7809118325350364, 0.8380007089684509, 0.7476267239835214, 0.6632796075683252, 0.7054999121419786, 0.7975601131541725, 0.8145601974264057, 0.7441281138790036, 0.7621487895387877, 0.7657342657342657, 0.7655645444952515, 0.8315512708150745, 0.7358655697531945, 0.7748017621145374, 0.8196548080309969, 0.7877457865168539, 0.7508021390374332, 0.740252803987894, 0.7863636363636364, 0.7867711053089643, 0.7833690221270521, 0.7460429124164615, 0.7573185518794388, 0.7876997915218902, 0.8155579021471313, 0.715917076598735]]
ACCURACY FINAL TEST 0:  0.7703162211918076
ACCURACY FINAL top3  TEST 0:  0.8818289587160325
'''



if __name__ == '__main__':
    # first_path = 'saved_data/DB1/' + window_path + '/' + window_path + '_exercise1_jitr_norm/subject_' + str(
    #     1)
    # directory = first_path + '/train'
    # X_TCN_fine_tune_train = np.load(directory + 'X_train.npy', encoding="bytes", allow_pickle=True)
    # print(X_TCN_fine_tune_train.shape)
    # plot_and_save_emg(X_TCN_fine_tune_train[0],'wavelet_images/rawSemg.pdf')
    # save_images(X_TCN_fine_tune_train[0],'wavelet_images')
    print("++++++++++++")
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []

    test_0 = []
    top3_test_0 = []
    test_1 = []

    for i in range(0, 1):
        accuracy_test_0,top3= calculate_fitness(i)
        print(accuracy_test_0)

        test_0.append(accuracy_test_0)
        top3_test_0.append(top3)

    print("ACCURACY FINAL TEST 0: ", test_0)
    print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
    print("ACCURACY FINAL top3  TEST 0: ", np.mean(top3_test_0))

    with open("../Pytorch_results_4_cycles.txt", "a") as myfile:
        myfile.write("CNN STFT: \n\n")
        myfile.write("Test 0: \n")
        myfile.write(str(np.mean(test_0, axis=0)) + '\n')
        myfile.write(str(np.mean(test_0)) + '\n')

        myfile.write("Test 1: \n")
        myfile.write(str(np.mean(test_1, axis=0)) + '\n')
        myfile.write(str(np.mean(test_1)) + '\n')
        myfile.write("\n\n\n")


