import csv
import os

import numpy as np

import nina_helper
import pickle

db1_path= '../../NinaproDB5/'
window_len=52
window_inc=5
window_path=str(window_len)+'_'+str(window_inc)
for i in range(1,11):
    data_dict = nina_helper.import_db5(db1_path, i, rest_length_cap=5)
    directory ='saved_data/DB5/'+window_path+'/'+window_path+'_exercise1_jitr_norm/data_dict_'
    save_path =directory+str(i)+'.pkl'
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    #保存到文件
    with open(save_path, 'wb') as file:
        pickle.dump(data_dict, file)