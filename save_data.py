import pickle
import os
import numpy as np
import load_evaluation_dataset, \
    load_pre_training_dataset,create_diversify_dataset

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5
#52窗口 5的间隔
def save_training_data_52_5(number_of_vector_per_example = 52,
number_of_canals = 8,
number_of_classes = 7,
size_non_overlap = 5 ):
    """训练集保存,所有角色的训练集"""
    examples, labels ,data_diversify= load_evaluation_dataset.read_data('../../EvaluationDataset','training0',number_of_vector_per_example = number_of_vector_per_example,
number_of_canals = number_of_canals,
number_of_classes = number_of_classes,
size_non_overlap = size_non_overlap)
    # 加入保存字典的代码
    # with open('saved_data/my_dict.pickle', 'wb') as file:
    #     pickle.dump(mydict, file)
    directory = 'saved_data/52_5_shift_electrodes'
    filename = 'saved_training_dataset.npy'
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 拼接目录和文件路径
    file_path = os.path.join(directory, filename)
    datasets = [examples, labels, data_diversify]
    np.save(file_path, datasets)
    """Test0测试集保存"""
    examples, labels, data_diversify = load_evaluation_dataset.read_data('../../EvaluationDataset', 'Test0',
                                                                         number_of_vector_per_example=number_of_vector_per_example,
                                                                         number_of_canals=number_of_canals,
                                                                         number_of_classes=number_of_classes,
                                                                         size_non_overlap=size_non_overlap)
    # 加入保存字典的代码
    # with open('saved_data/my_dict.pickle', 'wb') as file:
    #     pickle.dump(mydict, file)

    filename = 'saved_Test0_dataset.npy'
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 拼接目录和文件路径
    file_path = os.path.join(directory, filename)
    datasets = [examples, labels, data_diversify]
    np.save(file_path, datasets)
    """Test1测试集保存"""
    examples, labels, data_diversify = load_evaluation_dataset.read_data('../../EvaluationDataset', 'Test1',
                                                                         number_of_vector_per_example=number_of_vector_per_example,
                                                                         number_of_canals=number_of_canals,
                                                                         number_of_classes=number_of_classes,
                                                                         size_non_overlap=size_non_overlap)
    # 加入保存字典的代码
    # with open('saved_data/my_dict.pickle', 'wb') as file:
    #     pickle.dump(mydict, file)

    filename = 'saved_Test1_dataset.npy'
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 拼接目录和文件路径
    file_path = os.path.join(directory, filename)
    datasets = [examples, labels, data_diversify]
    np.save(file_path, datasets)

#150的窗口，50的间隔
def save_pretraining_data_100_50(number_of_vector_per_example = 150,
number_of_canals = 8,
number_of_classes = 7,
size_non_overlap = 50 ):

    #生成训练数据集产生文件
    examples, labels ,data_diversify= load_evaluation_dataset.read_data('../../EvaluationDataset',number_of_vector_per_example = number_of_vector_per_example,
number_of_canals = number_of_canals,
number_of_classes = number_of_classes,
size_non_overlap = size_non_overlap)
    # 加入保存字典的代码
    # with open('saved_data/my_dict.pickle', 'wb') as file:
    #     pickle.dump(mydict, file)
    datasets = [examples, labels,data_diversify]
    np.save("saved_data/saved_pre_training_dataset_100_50.npy", datasets)
def save_eval_data():
    #生成评估训练集产生文件
    examples, labels = load_evaluation_dataset.read_data('../../EvaluationDataset',
                                                         type='Test1')

    datasets = [examples, labels]
    pickle.dump(datasets, open("data/saved_dataset_test1.p", "wb"))

    examples, labels = load_evaluation_dataset.read_data('../../EvaluationDataset',
                                                         type='training0')

    datasets = [examples, labels]
    pickle.dump(datasets, open("data/saved_dataset_training.p", "wb"))

    examples, labels = load_evaluation_dataset.read_data('../../EvaluationDataset',
                                                         type='Test0')

    datasets = [examples, labels]
    pickle.dump(datasets, open("data/saved_dataset_test0.p", "wb"))

if __name__ == '__main__':
    save_training_data_52_5(number_of_vector_per_example,
number_of_canals ,
number_of_classes ,
size_non_overlap)
    # save_diversify_data()
    # save_pretraining_data_100_50()
    # save_pretraining_data()
    # import os
    #
    # print(os.listdir("../"))
    #
    # datasets_training = np.load("saved_data/saved_pre_training_dataset.npy", encoding="bytes", allow_pickle=True)
    # examples_training, labels_training = datasets_training
    # # print(examples_training)
