from matplotlib import pyplot as plt
import numpy as np
import csv
import re
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

thre = 2  # 要调整的参数,这个是阈值,决定是2σ还是3σ
iteration_num = 2  # 要调整的参数，这个是迭代次数


def data_check(original_data, final_processed_data):
    plt.figure()
    plt.title(u"4个UWB锚点采用离群点检测与卡尔曼滤波的测量结果(未去除相似点)")
    plt.xlabel(u"数据序号")
    plt.ylabel(u"测量值")
    for i in range(4):
        plt.plot(np.arange(len(original_data[i])), original_data[i], label="A" + str(i) + u"原始数据", linestyle='dashed')
        plt.plot(np.arange(len(final_processed_data[i])), final_processed_data[i], label="A" + str(i) + u"处理后的数据")

    plt.legend(loc="lower right")
    plt.show()


def save_data(file_path: str, Measures):
    with open(file_path, "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(["Number", "A0", "A1", "A2", "A3"])
        csvwriter.writerows(np.column_stack((np.arange(Measures.shape[1]), Measures.T)), )


def KalmanFilter(z, n_iter=20):
    # 卡尔曼滤波
    # 这里是假设A=1，H=1的情况

    # intial parameters
    sz = (n_iter,)  # size of array

    # Q = 1e-5 # process variance
    Q = 1e-6  # process variance
    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = 0.015 ** 2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    A = 1
    H = 1

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat


def read_txt(file_path: str):
    with open(file_path, "r") as f:  # 打开文件
        f.readline()  # 去掉第一行
        data = f.readlines()  # 读取文件

    data_num = len(data) / 4
    if int(data_num) - data_num < -0.1:
        raise ValueError("数据数量不对!")

    initial_time = re.search(":.*:([0-9]*)", data[0], flags=0)  # 获取初始数据序列
    initial_time = int(initial_time.group(1))
    Measures = []
    for i in range(int(data_num)):
        measure = []
        for j in range(4):
            device = []
            anchor = re.search(":[0-9]*?:RR:0:([0-9]):[0-9]*?:([0-9]*?):[0-9]*?:([0-9]*)", data[4 * i + j], flags=0)
            device.extend([int(anchor.group(3)) - initial_time, anchor.group(1), anchor.group(2)])  # 获取数据序号、设备号、测量值
            device = list(map(int, device))
            measure.append(device)  # 一个measure就是四个设备拿到的四份数据
        Measures.append(measure)
    Measures = np.array(Measures)  # Measures是三维数组是获取的所有测量数据
    return Measures


def remove_outliner(file_path: str):
    Measures = read_txt(file_path)
    normalized_device_data = []
    normalized_device_data_x = []
    device_data = []
    device_data_x = []
    for i in range(4):
        device_data.append(Measures[:, i, 2])
        device_data_x.append(np.arange(len(Measures[:, i, 2])))
        normalized_device_data.append(device_data[i] / np.max(Measures[:, i, 2]))  # 最大值归一化
    normalized_device_data_x = device_data_x
    normalized_device_data = np.array(normalized_device_data)
    normalized_device_data_x = np.array(normalized_device_data_x)
    device_data = np.array(device_data)
    device_data_x = np.array(device_data_x)
    processed_device_data = np.array(device_data).copy()
    device_mean = np.mean(device_data, axis=1)
    device_std = np.std(device_data, axis=1)

    low_thre = device_mean - device_std * thre  # 去除离群点
    high_thre = device_mean + device_std * thre  # 去除离群点

    for _ in range(iteration_num):
        for i in range(4):
            for j in range(len(device_data[i, :])):
                if device_data[i, j] < low_thre[i] or device_data[i, j] > high_thre[i]:
                    subtract_array = np.array(processed_device_data[:, j])
                    processed_device_data[i, j] = (np.sum(processed_device_data, axis=1)[i] - subtract_array[i]) \
                                                  / (processed_device_data.shape[1] - 1)  # 去除离群点后求均值
                    processed_device_mean = np.mean(processed_device_data, axis=1)
                    processed_device_std = np.std(processed_device_data, axis=1)
                    low_thre = processed_device_mean - processed_device_std * thre  # 更新阈值
                    high_thre = processed_device_mean + processed_device_std * thre  # 更新阈值
    return device_data, processed_device_data


def data_process(file_path: str):
    device_data, processed_device_data = remove_outliner(file_path)
    xhat = []
    for i in range(4):
        # raw_data = device_data[i]
        raw_data = processed_device_data[i]
        xhat.append(KalmanFilter(raw_data, n_iter=len(raw_data)))
    xhat = np.array(xhat)
    #xhat = np.around(xhat, 1)  # 将滤波后的四组坐标值，保留一位小数
    return device_data, xhat  # device_data为原始数据,xhat是离群点去除且卡尔曼滤波后的数据


def clean_similar_data(final_processed_data, accuracy=10):
    temp = final_processed_data[:, 0]
    dissimilar_data = temp
    for i in range(1, final_processed_data.shape[1] - 1):
        distance = abs(final_processed_data[:, i + 1] - temp)
        if distance[0] > accuracy or distance[1] > accuracy or distance[2] > accuracy or distance[3] > accuracy:
            dissimilar_data = np.row_stack((dissimilar_data, final_processed_data[:, i + 1]))
            temp = final_processed_data[:, i + 1]
    dissimilar_data = dissimilar_data.T
    return dissimilar_data


def calculate_num_of_dissimilar_data(save_path):
    normal_data = []
    abnormal_data = []
    for i in range(1, 325):
        file_path = "./data/附件1：UWB数据集/正常数据/" + str(i) + ".正常.txt"
        original_data, final_processed_data = data_process(file_path)
        dissimilar_data = clean_similar_data(final_processed_data, accuracy=10)
        normal_data.append(dissimilar_data.shape[1])

    for i in range(1, 325):
        file_path = "./data/附件1：UWB数据集/异常数据/" + str(i) + ".异常.txt"
        original_data, final_processed_data = data_process(file_path)
        dissimilar_data = clean_similar_data(final_processed_data, accuracy=10)
        abnormal_data.append(dissimilar_data.shape[1])

    with open(save_path + "/正常数据/清洗后的正常数据的数据数量.csv", "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(["Document Number", "The Number of Dissimilar Data"])
        for i in range(1, 325):
            csvwriter.writerow([str(i) + ".normal_document", normal_data[i - 1]])

    with open(save_path + "/异常数据/清洗后的异常数据的数据数量.csv", "w+", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(["Document Number", "The Number of Dissimilar Data"])
        for i in range(1, 325):
            csvwriter.writerow([str(i) + ".abnormal_document", abnormal_data[i - 1]])


def collect_dataset(kind, use_similar_data=True, only_remove_outliner=False):
    for i in range(1, 325):
        file_path = f"./data/附件1：UWB数据集/{kind}数据/{i}.{kind}.txt"
        if only_remove_outliner == False:
            original_data, final_processed_data = data_process(file_path)
            if use_similar_data:
                save_data(f"cleaned_data/{kind}数据/{i}.{kind}.csv", final_processed_data)
            else:
                dissimilar_data = clean_similar_data(final_processed_data, accuracy=10)  # 不去除相似点，让随机森林的训练资料更丰富
                save_data(f"cleaned_data/{kind}数据/{i}.{kind}.csv", dissimilar_data)
        else:
            original_data, final_processed_data = remove_outliner(file_path)
            save_data(f"remove_outliner_data/{kind}数据/{i}.{kind}.csv", final_processed_data)



if __name__ == '__main__':
    #collect_dataset("正常", only_remove_outliner=False,use_similar_data=False)  # 收集训练集用，特意为未去除相似点
    #collect_dataset("异常", only_remove_outliner=False,use_similar_data=False)
    file_path = ["data/附件1：UWB数据集/正常数据/24.正常.txt", "data/附件1：UWB数据集/正常数据/109.正常.txt", "data/附件1：UWB数据集/异常数据/1.异常.txt",
                 "data/附件1：UWB数据集/异常数据/100.异常.txt"]
    save_path = ["submit/task1/正常数据/24.正常(已清洗且去除相似点).csv", "submit/task1/正常数据/109.正常(已清洗且去除相似点).csv",
                 "submit/task1/异常数据/1.异常(已清洗且去除相似点).csv",
                 "submit/task1/异常数据/100.异常(已清洗且去除相似点).csv"]
    for i in range(4):
        original_data, final_processed_data = data_process(file_path[i])
        dissimilar_data = clean_similar_data(final_processed_data, accuracy=10)
        save_data(save_path[i], dissimilar_data)
        data_check(original_data, final_processed_data)

    calculate_num_of_dissimilar_data("./submit/task1")
