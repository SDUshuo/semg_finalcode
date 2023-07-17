import numpy as np
import pywt
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
该文件名为calculate_wavelet.py,主要包含三个函数:
1.函数名:calculate_wavelet_dataset(dataset) ,功能是将一个数据集中的每个数据的每个通道的电极向量经过小波变换处理后返回其系数。
该函数调用了calculate_wavelet_vector()函数和zoom（）函数。其中calculate_wavelet_vector()函数将电极向量用小波基函数进行小波变换，并返回系数，zoom（）函数用于对系数进行缩小处理。
1. 函数名：calculate_wavelet_vector(vector, mother_wavelet='mexh', scales=np.arange(1, 32))，该函数将传入的电极向量使用小波基函数进行小波变换，并返回系数。
2. 函数名: show_wavelet(coef) ,该函数实现了将小波变换后得到的系数绘制出来的功能，并显示出来。
"""


"""
维度变化的意义如下：

dataset：

维度：(N, M, L)
N表示数据集中示例的数量，M表示示例的通道数，L表示每个通道的样本数（时间戳大小=52）。
dataset是一个三维数组，其中每个示例都是一个二维数组，表示不同通道上的电极向量数据。

canals：
维度：(M', P, Q)
M'表示示例中的通道数，P表示小波系数矩阵的行数，Q表示小波系数矩阵的列数。
canals是一个三维数组，其中每个通道包含一个小波系数矩阵，表示在不同尺度和时间位置上的频率特征。

example_to_classify：
维度：(P, M', Q)
P表示小波系数矩阵的行数，M'表示示例中的通道数，Q表示小波系数矩阵的列数。
example_to_classify是一个三维数组，表示将通道和小波系数矩阵的维度进行交换后的结果。

dataset_spectrogram：
维度：(N, P, M', Q)
N表示数据集中示例的数量，P表示小波系数矩阵的行数，M'表示示例中的通道数，Q表示小波系数矩阵的列数。
dataset_spectrogram是一个四维数组，包含了数据集中每个示例的小波谱图。
综上所述，通过对数据集中的每个示例进行小波变换，代码生成了小波系数矩阵，并对通道和小波系数矩阵的维度进行了交换，最终得到了数据集的小波谱图表示。这种表示可以更好地捕捉信号在不同尺度和时间位置上的频率特征，为后续的信号处理和分类任务提供了有用的信息
"""
def calculate_wavelet_dataset(dataset):

    dataset_spectrogram = []
    # dataset_diversifydata=[]
    mother_wavelet = 'mexh'
    # dataset(N, M, L)  对每个examples，都是一个M*L，行指的是不同通道，列是时间戳，每一行的数据指的是这个通道的不同时间戳构成的一个时序信号
    for examples in dataset:
        canals = []
        # 对每个信号单独处理，这里是对一个大小为M*L的时间窗内的信号，对每一个通道的信号分别进行CWT
        for electrode_vector in examples:
            coefs = calculate_wavelet_vector(np.abs(electrode_vector), mother_wavelet=mother_wavelet, scales=np.arange(1, 33))  # 33 originally
            # print(np.shape(coefs)) #(7, 12)  (本来是52个采样点的t，经过变换变成了12）
            # show_wavelet(coef=coefs)
            '''
            对计算得到的小波系数进行处理。首先使用 zoom 函数将系数缩小为原来的四分之一，然后使用 np.delete 函数删除最后一行和最后一列。
            将处理后的小波系数数组 coefs 沿着第一个和第二个轴进行转置，即交换维度 0 和 1。
            '''
            coefs = zoom(coefs, .25, order=0)  #(12,8,7)
            # coefs = zoom(coefs, 17/52, order=0)  #(16, 10, 9)
            coefs = np.delete(coefs, axis=0, obj=len(coefs)-1)
            coefs = np.delete(coefs, axis=1, obj=np.shape(coefs)[1]-1)
            #行是时间戳，列是CWT的尺度列表
            canals.append(np.swapaxes(coefs, 0, 1))

        example_to_classify = np.swapaxes(canals, 0, 1)#(12.8.7)
        # print(example_to_classify.shape)
        # example_to_classify_str = np.array_str(example_to_classify)
        # my_dict[example_to_classify_str] = examples
        # dataset_diversifydata.append(examples)
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram

def calculate_wavelet_vector(vector, mother_wavelet='mexh', scales=np.arange(1, 32)):
    coef, freqs = pywt.cwt(vector, scales=scales, wavelet=mother_wavelet)
    return coef
"""
函数calculate_wavelet_vector接受三个参数：

vector：输入的一维向量。
mother_wavelet（默认值为'mexh'）：选择用于小波变换的母小波函数，可以是预定义的小波函数名（如'morl'、'haar'、'db2'等），也可以是自定义的小波函数。
scales（默认值为从1到32的一系列值）：指定要使用的尺度值，用于调整小波函数的频率范围。
函数内部通过调用pywt.cwt函数来进行连续小波变换。pywt.cwt函数接受三个参数：

vector：输入的一维向量。
scales：尺度值，用于调整小波函数的频率范围。
wavelet：母小波函数，可以是预定义的小波函数名或自定义的小波函数。
调用pywt.cwt函数后，返回值coef是一个二维数组，表示计算得到的小波系数矩阵。其中，每一行对应一个尺度值，每一列对应输入向量的一个元素位置。小波系数表示了在不同尺度和位置上的信号变化情况。

最后，函数返回小波系数矩阵coef作为结果。
"""
"""
在连续小波变换（CWT）中，输入向量是一维的，而输出的小波系数矩阵是二维的。这涉及到CWT的工作原理和对信号的分析方式。

CWT通过将输入信号与不同尺度和频率的小波函数进行卷积来分析信号的局部频率特征。输入信号的每个样本点都与小波函数进行卷积，得到相应的小波系数。

维度变化的意义如下：

输入向量：

维度：(N,)
每个元素表示输入信号在时间轴上的一个样本点。
小波系数矩阵（输出）：

维度：(M, N)
M表示选择的尺度数，N表示输入向量的长度。
每一行表示对应尺度下的小波系数，每一列表示输入信号在不同时间位置的小波系数。
小波系数表示了在不同尺度和时间位置上的信号局部频率特征。
维度变化的意义在于提供了关于信号在不同尺度和时间位置上的频率信息。小波系数矩阵的行数M代表了选择的尺度数，这是一个重要的参数，可以控制分析的频率范围。
每个尺度对应一种频率带宽，因此可以通过观察小波系数矩阵的不同行来了解信号在不同频率带宽上的特征。

另外，小波系数矩阵的列数N表示输入信号的长度，通过观察不同列的小波系数可以了解信号在不同时间位置上的频率特征随时间的变化情况。

综上所述，输出的二维小波系数矩阵提供了对输入信号在不同尺度和时间位置上频率特征的详细分析，为信号处理和分析提供了丰富的信息。
"""
def show_wavelet(coef):
    print(np.shape(coef))
    plt.rcParams.update({'font.size': 36})
    plt.matshow(coef)
    plt.ylabel('Scale')
    plt.xlabel('Samples')
    plt.show()

