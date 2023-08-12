import csv
import os

import numpy as np

import nina_helper
import pickle

db1_path= '../../DB1/'

# window_len=52
# window_inc=5
window_len=52
window_inc=5
window_path=str(window_len)+'_'+str(window_inc)
for i in range(0,27):
    data_dict = nina_helper.import_db1(db1_path, i+1, rest_length_cap=5)
    #1和2我给倒过来了
    directory ='saved_data/DB1/'+window_path+'/'+window_path+'_exercise1_jirm_relax_exer2/data_dict_'
    save_path =directory+str(i)+'.pkl'
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    #保存到文件ss
    with open(save_path, 'wb') as file:
        pickle.dump(data_dict, file)

# 从文件中读取
# save_path ='saved_data/DB1/200_50/data_dict_'+str(1)+'.pkl'
# with open(save_path, 'rb') as file:
#     data_dict = pickle.load(file)
# # data_dict = nina_helper.import_db1(db1_path, 1, rest_length_cap=5)
# reps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# nb_test_reps = 2
# base = [1, 2]
# train_reps, test_reps = nina_helper.gen_split_balanced(reps, nb_test_reps, base)
#
# emg_normalized  = nina_helper.normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :])
#
# # Generate training data
# X_train, Y_train, R_train = nina_helper.get_windows(train_reps[0, :], window_len, window_inc, emg_normalized, data_dict["move"], data_dict["rep"])
#
# # Generate testing data
# X_test, Y_test, R_test =  nina_helper.get_windows(test_reps[0, :], window_len, window_inc, emg_normalized, data_dict["move"], data_dict["rep"])
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) #(71129, 52, 10, 1) (71129,) (18579, 52, 10, 1) (18579,)
# filename = "label.csv"
#
# # 打开CSV文件并写入数据
# with open(filename, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(Y_train)
#
# print("列表已成功保存到CSV文件。")
# print(Y_train)