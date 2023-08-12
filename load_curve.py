import random

import numpy as np
from matplotlib import pyplot as plt

import Wavelet_CNN_Source_Network as Wavelet_CNN_Source_Network
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from torch.nn import functional as F
from scipy.stats import mode
import db_one_model as db_one_model
import LMF as LMFmodule
import TCN as TCNmodule
import myTRN as myTRN
import TRNmodule
import params_contact
from save_data import number_of_vector_per_example, number_of_classes, size_non_overlap, number_of_canals
import os
from load_evaluation_dataset_DB1 import newpath, window_inc, window_len, window_path
from sklearn.metrics import confusion_matrix
import seaborn as sns
number_of_class = 18
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
batchsize = 512

epochs = 30
lr = 0.01



def confusion_matrix(pred, Y, number_class=18):
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
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def calculate_fitness(seedlist):
    accuracy_test0 = []  # 存储测试集0的准确率
    top3_accuracy_test0=[]
    seed = 25
# 4-82.651  3-
    # #创建slowfushion
    # CNN = Wavelet_CNN_Source_Network.Net(number_of_class=7, batch_size=batchsize, number_of_channel=12,
    #                                      learning_rate=0.0404709, dropout=.5).cuda()
    setup_seed(seed)
    model = db_one_model.Net(tcn_inputs_channal=10, number_of_classes=number_of_class)

    for dataset_index in range(1, 2):
        setup_seed(seed)
        # 在每次循环开始之前设置随机数种子
        first_path ='saved_data/DB1/' + window_path + '/' + window_path + '_exercise1_jitr_norm/subject_' + str(
            dataset_index)
        # 准备训练集
        # directory = 'saved_data/DB1/52_5/subject_' + str(dataset_index) + '/train'
        # directory = 'saved_data/DB1/52_5/52_5_exercise1/subject_'+ str(dataset_index) + '/train'
        directory = first_path + '/train'
        X_fine_tune_train = np.load(directory + 'X_train_CWT.npy', encoding="bytes", allow_pickle=True)
        Y_fine_tune_train = np.load(directory + 'Y_train.npy', encoding="bytes", allow_pickle=True)
        X_TCN_fine_tune_train = np.load(directory + 'X_train.npy', encoding="bytes", allow_pickle=True)
        # 维度为 (N, M, 10, 7)
        N = X_fine_tune_train.shape[0]
        M = X_fine_tune_train.shape[1]
        if M != 12:
            # 100-70=24
            indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
            # 使用这些索引从第二个维度选择数据
            X_fine_tune_train = X_fine_tune_train[:, indices, :, :]

        # 准备测试集0的数据
        # directory = 'saved_data/DB1/52_5/subject_' + str(dataset_index) + '/test'aile
        # directory =  'saved_data/DB1/52_5/52_5_exercise1/subject_'+ str(dataset_index) + '/test'
        directory = first_path + '/test'
        X_test_0 = np.load(directory + 'X_test_CWT.npy', encoding="bytes", allow_pickle=True)
        Y_test_0 = np.load(directory + 'Y_test.npy', encoding="bytes", allow_pickle=True)
        X_TCN_fine_tune_test = np.load(directory + 'X_test.npy', encoding="bytes", allow_pickle=True)
        # 维度为 (N, M, 10, 7)
        N = X_test_0.shape[0]
        M = X_test_0.shape[1]
        if M != 12:
            # 100-70=24
            indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
            # 使用这些索引从第二个维度选择数据
            X_test_0 = X_test_0[:, indices, :, :]
        # 打乱用于微调的训练数据

        # 打乱用于微调的训练数据
        X_fine_tune, Y_fine_tune, X_fine_tune_TCN = scramble(X_fine_tune_train, Y_fine_tune_train,
                                                             X_TCN_fine_tune_train)
        # X_fine_tune, Y_fine_tune, X_fine_tune_TCN = X_fine_tune_train, Y_fine_tune_train,X_TCN_fine_tune_train

        val_scale = 0.1
        # 划分验证集
        valid_examples = X_fine_tune[0:int(len(X_fine_tune) * val_scale)]
        labels_valid = Y_fine_tune[0:int(len(Y_fine_tune) * val_scale)]
        valid_examples_TCN = X_fine_tune_TCN[0:int(len(X_fine_tune_TCN) * val_scale)]

        X_fine_tune = X_fine_tune[int(len(X_fine_tune) * val_scale):]
        Y_fine_tune = Y_fine_tune[int(len(Y_fine_tune) * val_scale):]
        X_fine_tune_TCN = X_fine_tune_TCN[int(len(X_fine_tune_TCN) * val_scale):]

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

        """
        定义损失函数
        """
        criterion = nn.NLLLoss(size_average=False)
        precision = 1e-8

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 学习率调度器（scheduler）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)

        """
        进行训练
        """

        multi_model = train_model(model, criterion, optimizer, scheduler,
                                  dataloaders={"train": trainloader, "val": validationloader,"test": test_0_loader}, people=dataset_index,
                                  precision=precision)

        # print(multi_model)
        multi_model.eval()
        total = 0
        correct_prediction_test_0 = 0
        top3_correct_prediction_test_0=0
        all_predictions_test_0 = []
        all_ground_truth_test_0 = []
        """
        进行测试集测试
        由于测试集的 batch size 设置为1，每个样本都能独立地传递给模型，避免了批量归一化层等模型组件对批量大小的依赖。
        """
        for k, data_test_0 in enumerate(test_0_loader, 0):
            # get the inputs
            inputs_test_0, ground_truth_test_0, inputs_test_0_TCN = data_test_0
            inputs_test_0, ground_truth_test_0, inputs_test_0_TCN = Variable(inputs_test_0.cuda()), Variable(
                ground_truth_test_0.cuda()), Variable(inputs_test_0_TCN.cuda())

            concat_input_TCN = inputs_test_0_TCN
            concat_input_trn = inputs_test_0

            outputs_test_0 = multi_model(concat_input_trn, concat_input_TCN)
            _, predicted_top1 = torch.max(outputs_test_0.data, 1)
            _, predicted_top3 = outputs_test_0.data.topk(3, 1, True, True)
            # 在这里，收集所有的预测和真实标签
            all_predictions_test_0.extend(predicted_top1.cpu().numpy().tolist())
            all_ground_truth_test_0.extend(ground_truth_test_0.data.cpu().numpy().tolist())
            correct_prediction_test_0 += (predicted_top1.cpu().numpy() ==
                                          ground_truth_test_0.data.cpu().numpy()).sum()
            top3_correct_prediction_test_0 += sum(
                [ground_truth_test_0.data.cpu().numpy()[i] in predicted_top3.cpu().numpy()[i] for i in
                 range(len(predicted_top3))])

            total += ground_truth_test_0.size(0)  # 总样本数量
        # 在测试循环结束后，计算混淆矩阵并保存为 PDF
        cm_test_0 = confusion_matrix(all_ground_truth_test_0, all_predictions_test_0)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm_test_0, annot=True, fmt="d")
        plt.title('Confusion matrix for test')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig('wavelet_images/confusion_matrix_test.pdf', format='pdf')

        print("ACCURACY TEST_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))
        print("TOP-3 ACCURACY TEST_0 FINAL : %.3f %%" % (100 * float(top3_correct_prediction_test_0) / float(total)))

        accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
        top3_accuracy_test0.append(100 * float(top3_correct_prediction_test_0) / float(total))

    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())
    print("AVERAGE TOP-3 ACCURACY TEST 0 %.3f" % np.array(top3_accuracy_test0).mean())

    return accuracy_test0,top3_accuracy_test0

def train_model(model, criterion, optimizer, scheduler, dataloaders, people, num_epochs=epochs, precision=1e-8,
                justtest=False):
    since = time.time()
    best_loss = float('inf')
    # if people > 1:
    #     model_weights = torch.load('best_weights_source_wavelet_db1.pt')
    #     model.load_state_dict(model_weights)
    # 耐心值 控制早停 。
    # 当验证损失值连续若干个周期没有改善时，耐心值逐渐减少，如果达到了设定的耐心值阈值，就会停止训练。
    patience = 30
    patience_increase = 10
    if not justtest:
        # Initialize empty lists to store loss and accuracy values for training and testing
        train_loss_history, train_acc_history = [], []
        val_loss_history, val_acc_history = [], []
        for epoch in range(num_epochs):
            # 同样的操作，只是在每个epoch开始时初始化这些值为0
            train_loss, train_corrects, train_total = 0., 0, 0
            val_loss, val_corrects, val_total = 0., 0, 0
            epoch_start = time.time()
            # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)

            # 对于每个epoch都有train和val环节
            for phase in ['train', 'test','val']:
                if phase == 'train':
                    model.train(True)
                else:
                    model.eval()

                running_loss = 0.
                running_corrects = 0
                total = 0

                for i, data in enumerate(dataloaders[phase], 0):
                    # get the inputs
                    inputs, labels, TCN_input = data

                    inputs, labels, TCN_input = Variable(inputs.cuda()), Variable(labels.cuda().long()), Variable(
                        TCN_input.cuda())

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    if phase == 'train':
                        """
                        进行模型的训练
                        """
                        outputs = model(inputs, TCN_input)
                        # print(labels.shape)
                        _, predictions = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()
                        loss = loss.item()

                    elif phase == 'val':
                        # 根据类别调整，53
                        accumulated_predicted = Variable(torch.zeros(len(inputs), number_of_class)).cuda()
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
                            _, prediction_from_this_sub_network = torch.max(outputs.data,
                                                                            1)  # 返回最大值和对应的索引。这里只是用到了索引，即给他预测了第几类
                            # 根据每次子模型的预测结果，在accumulated_predicted中对应位置加1，实现预测结果的累加。
                            accumulated_predicted[range(len(inputs)),
                            prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                            total_sub_pass += 1
                        _, predictions = torch.max(accumulated_predicted.data, 1)
                        loss = loss_intermediary / total_sub_pass
                    else:
                        inputs_test_0, ground_truth_test_0, inputs_test_0_TCN = data
                        inputs_test_0, ground_truth_test_0, inputs_test_0_TCN = Variable(
                            inputs_test_0.cuda()), Variable(
                            ground_truth_test_0.cuda()), Variable(inputs_test_0_TCN.cuda())
                        concat_input_TCN = inputs_test_0_TCN
                        concat_input_trn = inputs_test_0

                        outputs_test_0 = model(concat_input_trn, concat_input_TCN)
                        _, predictions = torch.max(outputs_test_0.data, 1)
                        loss = criterion(outputs_test_0, labels).item()

                    # statistics
                    running_loss += loss  # 累积每个batch的损失值。
                    running_corrects += torch.sum(predictions == labels.data)  # 统计预测正确的样本数量
                    total += labels.size(0)  # 累积样本的总数。

                epoch_loss = running_loss / total  # 当前epoch的平均损失值。
                epoch_acc = running_corrects / total  # 当前epoch的准确率。
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                elif phase == 'test':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
                # Append the loss and accuracy values to the lists

                print('{} Loss: {:.8f} Acc: {:.8}'.format(
                    phase, epoch_loss, epoch_acc))

                # 90 85.4
                # deep copy the model
                if phase == 'val':
                    scheduler.step(epoch_loss)  # 根据当前epoch的验证损失值来调整学习率，使用预定义的学习率调整策略。
                    if epoch_loss + precision < best_loss:  # 当前验证损失值加上一个小的精度值小于最佳损失值best_loss
                        # print("New best validation loss:", epoch_loss)
                        best_loss = epoch_loss
                        # 在验证集上获得最佳损失时保存模型，并将其保存为文件
                        torch.save(model.state_dict(), 'best_weights_source_wavelet_db1_loadcurve.pt')
                        patience = patience_increase + epoch  # 更新耐心值为当前epoch加上预定义的耐心增加值。
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - epoch_start))
            if epoch > patience: #如果当前epoch大于耐心值，则跳出循环，结束训练过程
                break
        actual_epochs = len(train_loss_history)
        # 生成loss曲线
        plt.figure()
        print(type(train_loss_history))
        plt.plot(range(actual_epochs), torch.tensor(train_loss_history).cpu().tolist(), label='train_loss')
        plt.plot(range(actual_epochs), torch.tensor(val_loss_history).cpu().tolist(), label='test_loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('wavelet_images/loss.pdf')

        # 生成准确率曲线
        plt.figure()
        plt.plot(range(actual_epochs), torch.tensor(train_acc_history).cpu().tolist(), label='train_accuracy')
        plt.plot(range(actual_epochs), torch.tensor(val_acc_history).cpu().tolist(), label='test_accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('wavelet_images/accuracy.pdf')
        print()

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))
        # load best model weights
        model_weights = torch.load('best_weights_source_wavelet_db1_loadcurve.pt')
        model.load_state_dict(model_weights)
        model.eval()
        return model
    else:
        model_weights = torch.load('best_weights_source_wavelet_db1_loadcurve.pt')
        model.load_state_dict(model_weights)
        model.eval()
        return model


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
    top3_test_0=[]
    test_1 = []

    for i in range(0,1):
        accuracy_test_0,top3_accuracy_test0 = calculate_fitness(i)
        print(accuracy_test_0)
        print(top3_accuracy_test0)
        test_0.append(accuracy_test_0)
        top3_test_0.append(top3_accuracy_test0)


    print("ACCURACY FINAL TEST 0: ", test_0)
    print("ACCURACY FINAL TEST 0: ", np.mean(test_0))


    with open("../Pytorch_results_4_cycles.txt", "a") as myfile:
        myfile.write("CNN STFT: \n\n")
        myfile.write("Test 0: \n")
        myfile.write(str(np.mean(test_0, axis=0)) + '\n')
        myfile.write(str(np.mean(test_0)) + '\n')

        myfile.write("Test 1: \n")
        myfile.write(str(np.mean(test_1, axis=0)) + '\n')
        myfile.write(str(np.mean(test_1)) + '\n')
        myfile.write("\n\n\n")
