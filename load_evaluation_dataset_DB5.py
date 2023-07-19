import os
import pickle

import numpy as np
import calculate_wavelet
import calculate_wavelet_db1
import nina_helper
from tqdm import tqdm


# number_of_vector_per_example = 150
# number_of_canals = 8
# number_of_classes = 7
# size_non_overlap = 50 #滑动窗口间隔

def format_data_to_train(dataset_example_formatted):

    data_calculated = calculate_wavelet_db1.calculate_wavelet_dataset(dataset_example_formatted)  # 计算小波谱图
    return np.array(data_calculated)


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
def apply_jitter(X_train):
    for i in range(len(X_train)):
            X_train[i] = nina_helper.jitter(X_train[i])
    return X_train
window_len=52
window_inc=5
window_path=str(window_len)+'_'+str(window_inc)
newpath ='saved_data/DB5/' + window_path + '/' + window_path + '_exercise1_jitr_norm/subject_'
"""
_exercise1_jitr_norm训练及测试机都是用了家噪音和normlize  家噪音确实效果编号了
_exercise1只用了norm
_exercise1_4ge_dataAug  加入了四种数据增强的方式扩充数据及  都很差，不用了
_all_jitr_norm 全部联系
"""
def read_data(number_of_vector_per_example = 200,
size_non_overlap = 50):
    print("Reading Data")

    for i in tqdm(range(1, 11)):
        #这是之前加载的字典的数据
        directory = 'saved_data/DB5/' + window_path + '/' + window_path + '_exercise1_jitr_norm/data_dict_'
        save_path = directory + str(i) + '.pkl'
        with open(save_path, 'rb') as file:
            data_dict = pickle.load(file)
        # data_dict = nina_helper.import_db1(db1_path, 1, rest_length_cap=5)
        reps = np.array([1, 2, 3, 4, 5, 6])
        nb_test_reps = 1
        base = [3]
        train_reps, test_reps = nina_helper.gen_split_balanced(reps, nb_test_reps, base)
        #print(train_reps[0, :]) [ 1  3  4  6  7  8  9 10]
        emg_normalized = nina_helper.normalise_emg(data_dict['emg'], data_dict['rep'], reps)

        # Generate training data
        X_train, Y_train, R_train = nina_helper.get_windows(train_reps[0, :], number_of_vector_per_example, size_non_overlap, emg_normalized,
                                                            data_dict["move"], data_dict["rep"])
        X_train = np.transpose(np.squeeze(X_train), (0, 2, 1))
        apply_jitter(X_train)
        X_test, Y_test, R_test = nina_helper.get_windows(test_reps[0, :], number_of_vector_per_example, size_non_overlap, emg_normalized,
                                                         data_dict["move"], data_dict["rep"])
        apply_jitter(X_test)
        X_test = np.transpose(np.squeeze(X_test), (0, 2, 1))
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) #(22800, 10, 52) (22800,) (5718, 10, 52) (5718,)
        X_train_CWT = format_data_to_train(X_train)
        print("第"+str(i)+"人已经处理完毕，形状如下:")
        print(X_train_CWT.shape)
        print(X_train.shape)
        print(Y_train.shape)
        directory = newpath + str(i) + '/train'

        # 如果目录不存在，则创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 拼接目录和文件路径

        np.save(directory+'X_train_CWT.npy', X_train_CWT)
        np.save(directory+'Y_train.npy', Y_train)
        np.save(directory+'X_train.npy', X_train)

        """测试集保存"""
        X_test_CWT = format_data_to_train(X_test)
        directory = newpath+str(i)+'/test'
        # 如果目录不存在，则创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 拼接目录和文件路径
        np.save(directory + 'X_test_CWT.npy', X_test_CWT)
        np.save(directory + 'Y_test.npy', Y_test)
        np.save(directory + 'X_test.npy', X_test)

    print("Finished Reading Data")

def read_data_ss(number_of_vector_per_example = 200,
size_non_overlap = 50):
    print("Reading Data")

    for i in tqdm(range(2, 28)):
        #这是之前加载的字典的数据
        directory = 'saved_data/DB1/' + window_path + '/' + window_path + '_exercise1/data_dict_'
        save_path = directory + str(i) + '.pkl'
        with open(save_path, 'rb') as file:
            data_dict = pickle.load(file)
        # data_dict = nina_helper.import_db1(db1_path, 1, rest_length_cap=5)
        reps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        nb_test_reps = 2
        base = [2, 5]
        train_reps, test_reps = nina_helper.gen_split_balanced(reps, nb_test_reps, base)
        emg_normalized = nina_helper.normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :])
        #print(train_reps[0, :]) [ 1  3  4  6  7  8  9 10]
        emg_Jitter = nina_helper.Jitter(emg_normalized, data_dict['rep'], reps)
        emg_scale = nina_helper.Scale(emg_normalized, data_dict['rep'], reps)
        emg_mag_wrap = nina_helper.magnitude_warping(data_dict['emg'], data_dict['rep'], reps)
        emg_time_wrap = nina_helper.time_wraping(data_dict['emg'], data_dict['rep'], reps)

        print("emg_Jitter")
        print(emg_Jitter.shape)  #(142586, 10)
        # Generate training data
        X_train, Y_train, R_train = nina_helper.get_windows(train_reps[0, :], number_of_vector_per_example, size_non_overlap, emg_Jitter,
                                                            data_dict["move"], data_dict["rep"])
        # X_train2, Y_train2, R_train2 = nina_helper.get_windows(train_reps[0, :], number_of_vector_per_example,
        #                                                        size_non_overlap, emg_scale,
        #                                                        data_dict["move"], data_dict["rep"])
        # X_train3, Y_train3, R_train3 = nina_helper.get_windows(train_reps[0, :], number_of_vector_per_example,
        #                                                        size_non_overlap, emg_mag_wrap,
        #                                                        data_dict["move"], data_dict["rep"])
        # X_train4, Y_train4, R_train4 = nina_helper.get_windows(train_reps[0, :], number_of_vector_per_example,
        #                                                        size_non_overlap, emg_time_wrap,
        # #                                                        data_dict["move"], data_dict["rep"])
        # X_train=np.concatenate((X_train1,X_train2),axis=0)
        # Y_train=np.concatenate((Y_train1,Y_train2),axis=0)

        X_train = np.transpose(np.squeeze(X_train), (0, 2, 1))

        X_test, Y_test, R_test = nina_helper.get_windows(test_reps[0, :], number_of_vector_per_example,
                                                               size_non_overlap, emg_Jitter,
                                                               data_dict["move"], data_dict["rep"])
        # X_test2, Y_test2, R_test2 = nina_helper.get_windows(test_reps[0, :], number_of_vector_per_example,
        #                                                        size_non_overlap, emg_scale,
        #                                                        data_dict["move"], data_dict["rep"])
        # X_test3, Y_test3, R_test3 = nina_helper.get_windows(test_reps[0, :], number_of_vector_per_example,
        #                                                        size_non_overlap, emg_mag_wrap,
        #                                                        data_dict["move"], data_dict["rep"])
        # X_test4, Y_test4, R_test4 = nina_helper.get_windows(test_reps[0, :], number_of_vector_per_example,
        #                                                        size_non_overlap, emg_time_wrap,
        #                                                        data_dict["move"], data_dict["rep"])
        # X_test = np.concatenate((X_test1, X_test2), axis=0)
        # Y_test = np.concatenate((Y_test1, Y_test2), axis=0)
        X_test = np.transpose(np.squeeze(X_test), (0, 2, 1))

        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        #(22754, 10, 52) (22754,) (5718, 10, 52) (5718,)
        X_train_CWT = format_data_to_train(X_train)
        print("第"+str(i)+"人train已经处理完毕，形状如下:")
        print(X_train_CWT.shape)
        print(X_train.shape)
        print(Y_train.shape)
        #(22800, 10, 52) (22800,) (5718, 10, 52) (5718,)
        directory = newpath + str(i) + '/train'

        # 如果目录不存在，则创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 拼接目录和文件路径

        np.save(directory+'X_train_CWT.npy', X_train_CWT)
        np.save(directory+'Y_train.npy', Y_train)
        np.save(directory+'X_train.npy', X_train)

        """测试集保存"""
        X_test_CWT = format_data_to_train(X_test)

        directory = newpath+str(i)+'/test'
        # 如果目录不存在，则创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 拼接目录和文件路径
        np.save(directory + 'X_test_CWT.npy', X_test_CWT)
        np.save(directory + 'Y_test.npy', Y_test)
        np.save(directory + 'X_test.npy', X_test)

    print("Finished Reading Data")
#原来的备份

if __name__ == '__main__':
    read_data(number_of_vector_per_example = window_len,
size_non_overlap = window_inc)