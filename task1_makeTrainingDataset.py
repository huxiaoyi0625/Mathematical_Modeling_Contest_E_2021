import csv
import re
import numpy as np

thre = 1.5  # 要调整的参数,这个是阈值
iteration_num = 2  # 要调整的参数，这个是迭代次数

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


def data_process(file_path: str):
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
                    processed_device_data[i, j] = device_mean[i]
    xhat = []
    for i in range(4):
        # raw_data = device_data[i]
        raw_data = processed_device_data[i]
        xhat.append(KalmanFilter(raw_data, n_iter=len(raw_data)))
    xhat = np.array(xhat)
    xhat = np.around(xhat, 1)   # 将滤波后的四组坐标值，保留一位小数
    return device_data, xhat  # device_data为原始数据,xhat是离群点去除且卡尔曼滤波后的数据


def save_data(file_path: str, Measures):
    with open(file_path, "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(["Number", "A0", "A1", "A2", "A3"])
        csvwriter.writerows(np.column_stack((np.arange(Measures.shape[1]), Measures.T)), )

def collect_dataset(kind):
    for i in range(1, 325):
        file_path = f"./data/附件1：UWB数据集/{kind}数据/{i}.{kind}.txt"
        original_data, final_processed_data = data_process(file_path)
        save_data(f"cleaned_data/{kind}数据/{i}.{kind}.csv", final_processed_data)

def collect_labels():
    pass

if __name__ == '__main__':
    collect_dataset("正常")
    collect_dataset("异常")
