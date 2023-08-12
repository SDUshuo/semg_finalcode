import numpy as np
import calculate_wavelet
number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5
"""
1.格式化数据以供训练
2.将电极移位并建立相应的数据集以解决高峰值敏锐度问题
3.读取数据并为建立数据集做好准备。
它导入了以下模块和文件：
- numpy
- calculate_wavelet (自定义模块)
该程序中使用的函数有：
- format_data_to_train：格式化数据以供训练
- shift_electrodes：将电极移位
- read_data：读取数据并为建立数据集做好准备。
该程序主要应用于数据预处理，以便其能够在PyTorch实现中使用以进行模型训练。
"""

"""
vector_to_format: 输入的 EMG 数据向量。维度为 (n,)，其中 n 是数据向量的长度。

dataset_example_formatted: 存储格式化后的数据集。维度为 (m, x, y)，其中 m 是示例的数量，
x 是电极向量的长度，y 是示例中电极向量的数量。初始时，dataset_example_formatted 为空列表。

example: 存储一个示例的电极向量。维度为 (x, y)，其中 x 是电极向量的长度（即为时间戳的多少)，y 是示例中电极向量的数量。初始时，example 为空列表。

代码中的循环遍历输入的 EMG 数据向量，并逐步构建示例和格式化后的数据集。当 EMG 向量中的值达到 8 个时，将其作为电极向量添加到示例中。
当示例中的电极向量数达到要求时，进行转置和裁剪操作，然后将示例添加到格式化后的数据集中。
最后，调用 calculate_wavelet.calculate_wavelet_dataset 计算小波谱图，返回格式化后的数据集。

综上所述，该函数将输入的 EMG 数据向量转换为三维数据，其中第一维表示示例的数量，第二维表示电极向量的长度，第三维表示示例中电极向量的数量。
"""
def format_data_to_train(vector_to_format):
    dataset_example_formatted = []  # 存储格式化后的数据集
    example = []  # 存储一个示例的电极向量
    emg_vector = []  # 存储一个示例的 EMG 向量

    for value in vector_to_format:
        emg_vector.append(value)  # 将值添加到 EMG 向量中
        #
        if len(emg_vector) >= 8:  # 当 EMG 向量中的值达到 8 个时
            if example == []:
                example = emg_vector  # 将 EMG 向量作为示例的电极向量
            else:
                example = np.row_stack((example, emg_vector))  # 将电极向量添加到示例中
            emg_vector = []  # 重置 EMG 向量
            #
            if len(example) >= number_of_vector_per_example:  # 满足一个时间窗的时间戳长度了
                example = example.transpose()  # 转置示例，使电极向量成为列
                #每个时间窗取完后加入dataset里
                dataset_example_formatted.append(example)  # 将示例添加到格式化后的数据集中
                example = example.transpose()  # 恢复示例的原始维度
                example = example[size_non_overlap:]  # 从示例中移除重叠部分 （之前的那个时间窗已经取值完毕了，下一个时间窗example的开头把生效的那些去除

    data_calculated = calculate_wavelet.calculate_wavelet_dataset(dataset_example_formatted)  # 计算小波谱图
    return np.array(data_calculated)



def shift_electrodes(examples, labels):
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active cannals for those classes
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
    # Do the shifting only if the absolute mean is greater or equal to 0.75
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

"""
list_dataset的维度：

list_dataset是一个包含19个元素的列表（12个Male候选人和7个Female候选人）。
每个元素代表一个候选人的数据集。
每个候选人的数据集是一个列表，其中包含number_of_classes * 4个示例。
每个示例是一个三维数组，具有维度(m, x, y)，其中m是示例的数量，x是电极向量的长度，y是示例中电极向量的数量。
因此，list_dataset的维度是：(19, number_of_classes * 4, m, x, y)

list_labels的维度：

list_labels是一个包含19个元素的列表，与list_dataset的结构相对应。
每个元素代表一个候选人的标签集。
每个标签集是一个列表，其中包含number_of_classes * 4个标签数组。
每个标签数组的维度与相应的示例数组相同。
因此，list_labels的维度是：(19, number_of_classes * 4, m)
"""
def read_data(path,preOrEva):
    print("Reading Data")
    list_dataset = []  # 存储数据集
    list_labels = []  # 存储标签集
    Female =[7,2]
    Male =[12,15]
    index=0
    if preOrEva == 'Test0' or preOrEva=='Test1':
        index=1
    for candidate in range(Male[index]):#12 male
        labels = []  # 存储标签
        examples = []  # 存储示例
        #记录完整的七个手势五秒钟称为一个循环（cycle），四个循环为一个回合(round)（7*5*4=140s）
            #这里相当于对每个手势遍历分析
        #也就是说，每个Male下面的28个dat文件，每个文件代表当前这个手势记录的数据，一个手势一个文件，4*7=28，当前这个手势重复
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path+'/Male'+str(candidate)+'/'+preOrEva+'/classe_%d.dat' % i, dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            # (189,12,8,7)
            dataset_example = format_data_to_train(data_read_from_file)  # 格式化数据
            examples.append(dataset_example)  # 将示例添加到示例集中
            #广播，labels一维，全为class值
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))  # 构建标签
        examples, labels = shif