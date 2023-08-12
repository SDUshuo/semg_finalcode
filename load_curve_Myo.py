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
from scipy.stats import mode
import finalmodel
import LMF as LMFmodule
import TCN as TCNmodule
import myTRN as myTRN
import TRNmodule
from save_data import number_of_vector_per_example,number_of_classes,size_non_overlap,number_of_canals
import seaborn as sns
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
    model = finalmodel.Net(tcn_inputs_channal=8,number_of_classes=7)

    for dataset_index in range(0, 1):
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

        val_scale=0.01
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



        """
        定义损失函数
        """
        criterion = nn.NLLLoss(size_average=False)
        precision = 1e-8


        optimizer = optim.Adam(model.parameters(), lr=0.008)

        #学习率调度器（scheduler）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)


        """
        进行训练
        """

        multi_model = train_model(model,criterion, optimizer, scheduler,
                              dataloaders={"train": trainloader, "val": validationloader}, precision=precision)

        # print(multi_model)
        multi_model.eval()
        total = 0
        correct_prediction_test_0 = 0
        top3_correct_prediction_test_0 = 0
        all_predictions_test_0 = []
        all_ground_truth_test_0 = []

        """
        进行测试集测试
        由于测试集的 batch size 设置为1，每个样本都能独立地传递给模型，避免了批量归一化层等模型组件对批量大小的依赖。
        """
        for k, data_test_0 in enumerate(test_0_loader, 0):
            # get the inputs
            inputs_test_0, ground_truth_test_0,inputs_test_0_TCN = data_test_0
            inputs_test_0, ground_truth_test_0,inputs_test_0_TCN = Variable(inputs_test_0.cuda()), Variable(ground_truth_test_0.cuda()),Variable(inputs_test_0_TCN.cuda())

            #对trn测试
            # concat_input = inputs_test_0
            #对TCN测试

            concat_input_TCN = inputs_test_0_TCN
            concat_input_trn = inputs_test_0
            # for i in range(20):
            #     concat_input_trn = torch.cat([concat_input_trn, inputs_test_0])
            #     concat_input_TCN = torch.cat([concat_input_TCN, inputs_test_0_TCN])
            #将 inputs_test_0 连接（concatenate） 20 次，形成 concat_input，目的可能是扩充输入数据量，把test的数据一次20batchsize送进去。


            outputs_test_0 = multi_model(concat_input_trn,concat_input_TCN)
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
            total += ground_truth_test_0.size(0)# 总样本数量
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

#models=[CNN,TCN,LMF,TRN]
def train_model(model,criterion, optimizer, scheduler,dataloaders, num_epochs=60, precision=1e-8):
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
