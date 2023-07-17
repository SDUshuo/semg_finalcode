import pickle

import numpy as np
import calculate_wavelet

# number_of_vector_per_example = 150
# number_of_canals = 8
# number_of_classes = 7
# size_non_overlap = 50 #滑动窗口间隔
'''
原论文
'''

"""
这个程序文件加载预训练要用到的数据集，并准备分类数据以进行训练。它包括以下几个函数：
1. format_data_to_train 函数将 EMG 数据格式化为可以用于训练的数据。
2. shift_electrodes 函数将电极数据进行转换以适应训练数据，以便在不同的通道上进行分布式学习。
3. read_data 函数读取数据，格式化数据，调用 shift_electrodes 函数并最终返回处理过的数据。
"""

"""
vector_to_format: 输入的 EMG 数据向量。维度为 (n,)，其中 n 是数据向量的长度。
dataset_example_formatted: 存储格式化后的数据集。维度为 (m, x, y)，其中 m 是示例的数量，x 是电极向量的长度，y 是示例中电极向量的数量。初始时，dataset_example_formatted 为空列表。
example: 存储一个示例的电极向量。维度为 (x, y)，其中 x 是电极向量的长度，y 是示例中电极向量的数量。初始时，example 为空列表。
emg_vector: 存储一个示例的 EMG 向量。维度为 (x,)，其中 x 是电极向量的长度。初始时，emg_vector 为空列表。
"""


def format_data_to_train(vector_to_format,number_of_vector_per_example = 52,
number_of_canals = 8,
number_of_classes = 7,
size_non_overlap = 5):

    dataset_example_formatted = []  # 存储格式化后的数据集
    example = []  # 存储一个示例的电极向量
    emg_vector = []  # 存储一个示例的 EMG 向量
    for value in vector_to_format:
        emg_vector.append(value)  # 将值添加到 EMG 向量中
        if len(emg_vector) >= 8:  # 当 EMG 向量中的值达到 8 个时
            if example == []:
                example = emg_vector  # 将 EMG 向量作为示例的电极向量
            else:
                example = np.row_stack((example, emg_vector))  # 将电极向量添加到示例中
            emg_vector = []  # 重置 EMG 向量
            if len(example) >= number_of_vector_per_example:  # 当示例中的电极向量数达到要求时
                example = example.transpose()  # 转置示例，使电极向量成为列

                dataset_example_formatted.append(example)  # 将示例添加到格式化后的数据集中
                example = example.transpose()  # 恢复示例的原始维度
                example = example[size_non_overlap:]  # 从示例中移除重叠部分

    data_calculated,data_diversify = calculate_wavelet.calculate_wavelet_dataset(dataset_example_formatted)  # 计算小波谱图
    return np.array(data_calculated),np.array(data_diversify)


def shift_electrodes(examples, labels):
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.extend(examples[k])
            Y_example.extend(labels[k])

        cwt_add = []
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if cwt_add == []:
                    cwt_add = np.array(X_example[j][0])
                else:
                    cwt_add += np.array(X_example[j][0])
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0)))

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)):
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example


def read_data(path,type,number_of_vector_per_example = 52,
number_of_canals = 8,
number_of_classes = 7,
size_non_overlap = 5):
    print("Reading Data")
    list_dataset = []  # 存储数据集
    list_labels = []  # 存储标签集
    list_diversifydata=[]
#Male
    for candidate in range(16):
        labels = []  # 存储标签
        examples = []  # 存储示例
        diversify_data=[]
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path+'/Male'+str(candidate)+'/'+type+'/classe_%d.dat' % i, dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example, dataset_diversify = format_data_to_train(data_read_from_file,number_of_vector_per_example = number_of_vector_per_example,
number_of_canals = number_of_canals,
number_of_classes = number_of_classes,
size_non_overlap = size_non_overlap)  # 格式化数据
            examples.append(dataset_example)  # 将示例添加到示例集中
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))  # 构建标签
            diversify_data.append(dataset_diversify)
        examples, labels = shift_electrodes(examples, labels)  # 对示例进行电极转换
        list_dataset.append(examples)  # 将示例集添加到数据集中
        list_labels.append(labels)  # 将标签集添加到标签集中
        list_diversifydata.append(diversify_data)
#Female
    for candidate in range(2):
        labels = []  # 存储标签
        examples = []  # 存储示例
        diversify_data = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) +'/'+type+'/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example, dataset_diversify = format_data_to_train(data_read_from_file,number_of_vector_per_example = number_of_vector_per_example,
number_of_canals = number_of_canals,
number_of_classes = number_of_classes,
size_non_overlap = size_non_overlap)  # 格式化数据
            diversify_data.append(dataset_diversify)
            examples.append(dataset_example)  # 将示例添加到示例集中
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))  # 构建标签
        examples, labels = shift_electrodes(examples, labels)  # 对示例进行电极转换
        list_dataset.append(examples)  # 将示例集添加到数据集中
        list_labels.append(labels)  # 将标签集添加到标签集中
        list_diversifydata.append(diversify_data)


    print("Finished Reading Data")

    return list_dataset, list_labels,list_diversifydata
