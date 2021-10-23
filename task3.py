# -*- coding: utf-8 -*-
#这是任务3代码，为节省精力，基本都复制自任务2代码，仅修改了A0,A1等的位置，以及读取的txt文件

import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.optimize import root
import pandas as pd
import csv
from task1_plotFourPictures import read_txt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from task4_data_classification import read_txt
import joblib
from sklearn.neural_network import MLPRegressor

global d0, d1, d2, d3  # 即 A0,A1,A2,A3的测量值
global A0, A1, A2, A3
A0 = [0, 0, 1200]
A1 = [5000, 0, 1600]
A2 = [0, 3000, 1600]
A3 = [5000, 3000, 1200]

from matplotlib import pyplot


def show_3D_figure():
    # 函数未完成，不观察三维点数据了
    test_x = read_txt("./data/附件4：测试集.txt")
    test_x = test_x[:, :, 2]
    plt.Figure()
    ax = plt.axes(projection='3d')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('X')
    plt.ylabel('y')

    global d0, d1, d2, d3

    for i in range(test_x.shape[0]):
        size = 20 + 20 * np.random.rand()
        color = np.random.rand(3)
        d0 = test_x[i, 0]
        d1 = test_x[i, 1]
        d2 = test_x[i, 2]
        d3 = test_x[i, 3]
        tag = np.array(calculate_4_tag_position())
        for _ in range(3):
            ax.scatter3D(tag[i, 0], tag[i, 1], tag[i, 2], c=color, s=size, alpha=1)
        plt.show()


def A0_A1_A2_Positioning(Tag):
    x, y, z = Tag[0], Tag[1], Tag[2]
    return [
        (x - A0[0]) ** 2 + (y - A0[1]) ** 2 + (z - A0[2]) ** 2 - d0 ** 2,  # A0距离约束
        (x - A1[0]) ** 2 + (y - A1[1]) ** 2 + (z - A1[2]) ** 2 - d1 ** 2,  # A1距离约束
        (x - A2[0]) ** 2 + (y - A2[1]) ** 2 + (z - A2[2]) ** 2 - d2 ** 2,  # A2距离约束
    ]


def A0_A1_A3_Positioning(Tag):
    x, y, z = Tag[0], Tag[1], Tag[2]
    return [
        (x - A0[0]) ** 2 + (y - A0[1]) ** 2 + (z - A0[2]) ** 2 - d0 ** 2,  # A0距离约束
        (x - A1[0]) ** 2 + (y - A1[1]) ** 2 + (z - A1[2]) ** 2 - d1 ** 2,  # A1距离约束
        (x - A3[0]) ** 2 + (y - A3[1]) ** 2 + (z - A3[2]) ** 2 - d3 ** 2,  # A3距离约束
    ]


def A0_A2_A3_Positioning(Tag):
    x, y, z = Tag[0], Tag[1], Tag[2]
    return [
        (x - A0[0]) ** 2 + (y - A0[1]) ** 2 + (z - A0[2]) ** 2 - d0 ** 2,  # A0距离约束
        (x - A2[0]) ** 2 + (y - A2[1]) ** 2 + (z - A2[2]) ** 2 - d2 ** 2,  # A2距离约束
        (x - A3[0]) ** 2 + (y - A3[1]) ** 2 + (z - A3[2]) ** 2 - d3 ** 2,  # A3距离约束
    ]


def A1_A2_A3_Positioning(Tag):
    x, y, z = Tag[0], Tag[1], Tag[2]
    return [
        (x - A1[0]) ** 2 + (y - A1[1]) ** 2 + (z - A1[2]) ** 2 - d1 ** 2,  # A1距离约束
        (x - A2[0]) ** 2 + (y - A2[1]) ** 2 + (z - A2[2]) ** 2 - d2 ** 2,  # A2距离约束
        (x - A3[0]) ** 2 + (y - A3[1]) ** 2 + (z - A3[2]) ** 2 - d3 ** 2,  # A3距离约束
    ]


def read_last_data(file_path):
    data = pd.read_csv(file_path, encoding="gbk")
    last_distance = data.iloc[-1]
    last_distance = np.array(last_distance)
    global d0, d1, d2, d3
    d0, d1, d2, d3 = last_distance[1:5]  # 获取最后一行的A0-A3


def calculate_4_tag_position():
    tag = []
    tag.append(root(A0_A1_A2_Positioning, [2500, 1500, 1400]).x)  # 初始点选了4个anchor的中点
    tag.append(root(A0_A1_A3_Positioning, [2500, 1500, 1400]).x)  # 初始点选了4个anchor的中点
    tag.append(root(A0_A2_A3_Positioning, [2500, 1500, 1400]).x)  # 初始点选了4个anchor的中点
    tag.append(root(A1_A2_A3_Positioning, [2500, 1500, 1400]).x)  # 初始点选了4个anchor的中点
    return tag


def clustering_4_tag_position(cluster_tag):
    pass


def read_label(file_path):
    pass


def tag_check(cluster_tag, predicted_tag, actual_tag):
    pass


thre = 1.5  # 要调整的参数,这个是阈值
iteration_num = 2  # 要调整的参数，这个是迭代次数

'''
for _ in range(iteration_num):
    for i in range(4):
        for j in range(len(device_data[i, :])):
            if device_data[i, j] < low_thre[i] or device_data[i, j] > high_thre[i]:
                processed_device_data[i, j] = device_mean[i]
'''


def getData(kind):
    with open("submit/task2/predicted_data.csv", "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerow(["Number", "x1", "y1", "z1", "x", "y", "z"])

        correct_tag_position = pd.read_table("data/附件1：UWB数据集/Tag坐标信息.txt", delim_whitespace=True)  # 打开文件
        correct_tag_position = np.array(correct_tag_position.drop(columns=correct_tag_position.columns[0]))

        for index in range(1, 325):
            # data = pd.read_csv(f"cleaned_data/{kind}数据/{i}.{kind}.csv")
            # last_line = np.array(data.tail(1))
            read_last_data(f"cleaned_data/{kind}数据/{index}.{kind}.csv")
            cluster_tag = np.array(calculate_4_tag_position())  # 产生4个可行点，用于聚类
            cluster_tag_mean = cluster_tag.mean(axis=0)
            cluster_tag_std = cluster_tag.std(axis=0)
            low_thre = cluster_tag_mean - cluster_tag_std * thre  # 去除离群点
            high_thre = cluster_tag_mean + cluster_tag_std * thre  # 去除离群点
            for _ in range(iteration_num):
                for i in range(4):
                    for j in range(3):
                        if cluster_tag[i, j] < low_thre[j] or cluster_tag[i, j] > high_thre[j]:
                            cluster_tag[i, j] = cluster_tag_mean[j]

            predicted_tag = np.around(cluster_tag.mean(axis=0) / 10.0, 2)
            result = np.append(index, np.append(np.array(predicted_tag.T), np.array(correct_tag_position[index - 1])))
            csvwriter.writerow(result)


def array_add_same_rows(array, row, append_row_num):
    append_array = np.zeros((append_row_num, 3))
    for i in range(append_row_num):
        append_array[i, :] = row
    return np.row_stack((array, append_array))


def read_dataset(kinds=None):
    if kinds is None:
        kinds = ["正常", "异常"]
    data = []
    num_of_kind_data = []
    correct_tag_position = pd.read_table("data/附件1：UWB数据集/Tag坐标信息.txt", delim_whitespace=True)  # 打开文件
    correct_tag_position = np.array(correct_tag_position.drop(columns=correct_tag_position.columns[0]))
    for kind in kinds:
        #kind_data = np.array(pd.read_csv(f"remove_outliner_data/{kind}数据/{1}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]
        kind_data = np.array(pd.read_csv(f"cleaned_data/{kind}数据/{1}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]
        kind_label = array_add_same_rows(correct_tag_position[0, :], correct_tag_position[0, :],
                                         kind_data.shape[0] - 1)  # 制作三维标签
        for i in range(2, 325):
            #kind_data = np.row_stack((kind_data, np.array(
            #    pd.read_csv(f"remove_outliner_data/{kind}数据/{i}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]))
            kind_data = np.row_stack((kind_data, np.array(
                pd.read_csv(f"cleaned_data/{kind}数据/{i}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]))
            kind_label = array_add_same_rows(kind_label, correct_tag_position[i - 1, :],
                                             kind_data.shape[0] - kind_label.shape[0])
        if kind == "正常":
            normal_data = kind_data
            normal_label = kind_label
        elif kind == "异常":
            abnormal_data = kind_data
            abnormal_label = kind_label
    return (normal_data, abnormal_data, normal_label, abnormal_label)


def train_random_forest(data, label):
    # 建立随机森林

    X = data
    Y = label
    train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=0, test_size=0.2)

    rfc = RandomForestRegressor(n_estimators=20, random_state=90)
    rfc.fit(train_X, train_y)
    val_predictions = rfc.predict(val_X)
    print("mse:" + str(mean_squared_error(val_y, val_predictions)))
    print(np.sum((val_y - val_predictions) ** 2, axis=0) / val_y.shape[0])
    # 用交叉验证计算得分
    score_pre = cross_val_score(rfc, X, Y, cv=3).mean()
    print(f"交叉验证结果为:{score_pre}")
    return rfc


def save_dataset(dataset_x, label, dataset_file_path):
    with open(dataset_file_path[0], "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerows(dataset_x)
    with open(dataset_file_path[1], "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerows(label)


def make_sensor_fusion_dataset():
    normal_data, abnormal_data, normal_label, abnormal_label = read_dataset()
    global d0, d1, d2, d3
    normal_dataset_x = []
    for data_of_anchor in normal_data:
        d0, d1, d2, d3 = data_of_anchor
        tag = np.array(calculate_4_tag_position())
        normal_dataset_x.append(tag.flatten())
    normal_dataset_x = np.array(normal_dataset_x)

    abnormal_dataset_x = []
    for data_of_anchor in abnormal_data:
        d0, d1, d2, d3 = data_of_anchor
        tag = np.array(calculate_4_tag_position())
        abnormal_dataset_x.append(tag.flatten())
    abnormal_dataset_x = np.array(abnormal_dataset_x)

    normal_dataset_path = ["./sensor_fusion_data/normal_dataset.csv", "./sensor_fusion_data/normal_label.csv"]
    abnormal_dataset_path = ["./sensor_fusion_data/abnormal_dataset.csv", "./sensor_fusion_data/abnormal_label.csv"]
    save_dataset(normal_dataset_x, normal_label, normal_dataset_path)
    save_dataset(abnormal_dataset_x, abnormal_label, abnormal_dataset_path)


def read_sensor_fusion_dataset(dataset_path, label_path):
    dataset = pd.read_csv(dataset_path)  # 12列数据即对应着A0A1A2, A0A1A3, A0A2A3, A1A2A3这四种定位方式所计算的可能三维坐标
    label = pd.read_csv(label_path)  # 真实的三维数据
    return (np.array(dataset), np.array(label))


def save_forest_model():
    normal_dataset_file_path = ["./sensor_fusion_data/normal_dataset.csv", "./sensor_fusion_data/normal_label.csv"]
    abnormal_dataset_file_path = ["./sensor_fusion_data/abnormal_dataset.csv",
                                  "./sensor_fusion_data/abnormal_label.csv"]
    normal_dataset_x, normal_label = read_sensor_fusion_dataset(normal_dataset_file_path[0],
                                                                normal_dataset_file_path[1])
    normal_dataset_regression_rfc = train_random_forest(normal_dataset_x, normal_label)

    abnormal_dataset_x, abnormal_label = read_sensor_fusion_dataset(abnormal_dataset_file_path[0],
                                                                    abnormal_dataset_file_path[1])
    abnormal_dataset_regression_rfc = train_random_forest(abnormal_dataset_x, abnormal_label)
    joblib.dump(normal_dataset_regression_rfc, 'task2_normal_dataset_regression_rfc.model')
    joblib.dump(abnormal_dataset_regression_rfc, 'task2_abnormal_dataset_regression_rfc.model')

def training_MLP_Model():
    normal_dataset_file_path = ["./sensor_fusion_data/normal_dataset.csv", "./sensor_fusion_data/normal_label.csv"]
    abnormal_dataset_file_path = ["./sensor_fusion_data/abnormal_dataset.csv",
                                  "./sensor_fusion_data/abnormal_label.csv"]
    normal_dataset_x, normal_label = read_sensor_fusion_dataset(normal_dataset_file_path[0],
                                                                normal_dataset_file_path[1])
    abnormal_dataset_x, abnormal_label = read_sensor_fusion_dataset(abnormal_dataset_file_path[0],
                                                                    abnormal_dataset_file_path[1])

    X = normal_dataset_x
    y = normal_label
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.2)
    clf = MLPRegressor(solver='adam',
                        hidden_layer_sizes=(150, 95,50,30), random_state=1)

    clf.fit(train_X, train_y)
    val_predictions = clf.predict(val_X)
    #MSE=np.sum(abs(val_y - val_predictions)**2 ,axis=0) / val_y.shape[0]
    error = np.sum(abs(val_y - val_predictions) ,axis=0) / val_y.shape[0]
    print(mean_squared_error(val_y, val_predictions))
    joblib.dump(clf, 'task2_normal_dataset_regression_MLP.model')

    X = abnormal_dataset_x
    y = abnormal_label
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.2)
    clf = MLPRegressor(solver='adam',
                        hidden_layer_sizes=(600, 300,160,60), random_state=1)


    clf.fit(train_X, train_y)
    val_predictions = clf.predict(val_X)
    clf2= joblib.load('task2_normal_dataset_regression_MLP.model')
    val_predictions_2 = clf2.predict(val_X)
    print(mean_squared_error(val_y, val_predictions))
    error = np.sum(abs(val_y - val_predictions), axis=0) / val_y.shape[0]
    joblib.dump(clf, 'task2_abnormal_dataset_regression_MLP.model')

if __name__ == '__main__':
    #make_sensor_fusion_dataset() # 制作传感器融合数据集会花费大量时间,建议非必要不制作
    #training_MLP_Model()  # 训练神经网络
    #save_forest_model()  # 训练并保存模型

    normal_test_x = read_txt("./data/附件3：测试集（无干扰）.txt")
    normal_test_x = normal_test_x[:, :, 2]
    model = joblib.load("task2_normal_dataset_regression_MLP.model")

    predicted_normal_tag = []
    for i in range(normal_test_x.shape[0]):
        d0, d1, d2, d3 = normal_test_x[i, :]
        predicted_normal_tag.append(numpy.array(calculate_4_tag_position()).flatten())
    predicted_normal_tag = np.array(predicted_normal_tag)
    finally_predicted_normal_tag = model.predict(predicted_normal_tag)

    abnormal_test_x = read_txt("./data/附件3：测试集（无干扰）.txt")
    abnormal_test_x = abnormal_test_x[:, :, 2]
    model = joblib.load("task2_abnormal_dataset_regression_rfc.model")

    predicted_abnormal_tag = []
    for i in range(abnormal_test_x.shape[0]):
        d0, d1, d2, d3 = abnormal_test_x[i, :]
        predicted_abnormal_tag.append(numpy.array(calculate_4_tag_position()).flatten())
    predicted_abnormal_tag = np.array(predicted_abnormal_tag)
    finally_predicted_abnormal_tag = model.predict(predicted_abnormal_tag)

    with open("./submit/task3/predicted_position.csv", "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        csvwriter.writerows(finally_predicted_normal_tag)
        csvwriter.writerows(finally_predicted_abnormal_tag)
    breakpoint()
