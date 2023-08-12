import numpy as np
import calculate_wavelet
number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 10 #滑动窗口间隔
"""
对一个用户的class.dat文件按照时间窗读取，每个时间窗作为一个example，合起来作为dataset_example_formatted返回
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
                example = example.transpose()  # 转置示例，使t成为列
                #每个时间窗取完后加入dataset里
                # print(np.array(example))
                dataset_example_formatted.append(example)  # 将示例添加到格式化后的数据集中
                example = example.transpose()  # 恢复示例的原始维度
                example = example[size_non_overlap:]  # 从示例中移除重叠部分 （之前的那个时间窗已经取值完毕了，下一个时间窗example的开头把生效的那些去除

    return np.array(dataset_example_formatted)



"""
采用的第三种数据增强技术旨在模拟皮肤上的电极位移。这是特别有趣的，因为数据集是用干电极臂带记录的，这种噪音是可以预料的。
数据增强技术包括将部分功率谱幅度从一个通道移动到下一个通道。
换句话说，来自每个通道的部分信号能量被发送到相邻通道，模拟皮肤上的电极位移。在这项工作中，这种方法将被称为电极位移增强
"""
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
def read_data(path,preOrEva='training0'):
    print("Reading Data")
    list_dataset = []  # 存储数据集
    list_y = []  # 存储标签集
    Female =[10,2]
    Male =[14,15]
    index=0
    bianhao = 9
    y_candidate = []
    if preOrEva == 'Test0' or preOrEva=='Test1':
        index=1
    for candidate in range(Male[index]):#14 male
        labels = []  # 存储标签
        examples = []  # 存储示例
        peoples =[]
        domains=[]

        #记录完整的七个手势五秒钟称为一个循环（cycle），四个循环为一个回合(round)（7*5*4=140s）
            #这里相当于对每个手势遍历分析
        #也就是说，每个Male下面的28个dat文件，每个文件代表当前这个手势记录的数据，一个手势一个文件，4*7=28，当前这个手势重复
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path+'/Male'+str(candidate)+'/'+preOrEva+'/classe_%d.dat' % i, dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            # (189,8,t)
            dataset_example = format_data_to_train(data_read_from_file)  # 格式化数据

            #这是为了diversify构建 y 二维数组，时间窗个数*3  列中第一列为label第二列为人类编号第三列都是0
            label=(i % number_of_classes) + np.zeros(dataset_example.shape[0])
            people = candidate+ np.zeros(dataset_example.shape[0])
            domain =np.zeros(dataset_example.shape[0])
            y_candidate_class =np.transpose([label, people, domain])

            y_candidate.append(y_candidate_class)
            #对28个class文件获取的dataset_example加入examples中
            list_dataset.append(dataset_example)
            #广播，labels一维，全为class值，数量为时间窗个数189  例如【1，，，，1】
            #dataset_example.shape[0] 就是时间窗个数189

        # examples, labels = shift_electrodes(examples, labels)  # 对示例进行电极转换
        #这里是所有的28个文件的总和，也就是一个测试者的所有数据，labels有28个，每一个都是一个全为class值的一维向量，一维向量的列数和时间窗个数一致
        #examples也是28个，每一个都是一个(189,8，t)的数据 189是时间窗的个数（不一定就189），8是通道，t是时间窗采样点，

        # list_dataset.append(examples)  # 将示例集添加到数据集中
        #
        # list_labels.append(labels)  # 将标签集添加到标签集中

    '''
    diversify
    # 读取.npy文件
        datax = np.load('data/emg/emg_x.npy')  # (6883, 8, 200)
        data = np.load('data/emg/emg_y.npy')  # (6883, 3)
    '''

    for candidate in range(Female[index]):#10 female
        labels = []  # 存储标签
        examples = []  # 存储示例

        bianhao += 1
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) +'/'+preOrEva+'/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)  # 格式化数据
            # 这是为了diversify构建 y 二维数组，时间窗个数*3  列中第一列为label第二列为人类编号�