from matplotlib import pyplot as plt
from task1_plotFourPictures import read_txt
import numpy as np
from scipy.optimize import root
import joblib
from task1_plotFourPictures import KalmanFilter
'''
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
'''
global d0, d1, d2, d3  # 即 A0,A1,A2,A3的测量值
global A0, A1, A2, A3
A0 = [0, 0, 1300]
A1 = [5000, 0, 1700]
A2 = [0, 5000, 1700]
A3 = [5000, 5000, 1300]

def calculate_4_tag_position():
    tag = []
    tag.append(root(A0_A1_A2_Positioning, [2500, 2500, 1500]).x)  # 初始点选了4个anchor的中点
    tag.append(root(A0_A1_A3_Positioning, [2500, 2500, 1500]).x)  # 初始点选了4个anchor的中点
    tag.append(root(A0_A2_A3_Positioning, [2500, 2500, 1500]).x)  # 初始点选了4个anchor的中点
    tag.append(root(A1_A2_A3_Positioning, [2500, 2500, 1500]).x)  # 初始点选了4个anchor的中点
    return tag

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

def show_3D_figure(trjectory):
    # 函数未完成，不观察三维点数据了
    fig=plt.Figure()
    ax = plt.axes(projection='3d')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('X')
    plt.ylabel('y')
    plt.plot(trajectory[:,0],trjectory[:,1],trjectory[:,2],label="trajectory")
    plt.legend()
    plt.show()

def show_original_data(test_x):
    plt.figure()
    plt.plot(np.arange(test_x.shape[0]),test_x[:,0])
    plt.plot(np.arange(test_x.shape[0]), test_x[:, 1])
    plt.plot(np.arange(test_x.shape[0]), test_x[:, 2])
    plt.plot(np.arange(test_x.shape[0]), test_x[:, 3])
    plt.show()

def get_trajectory():
    test_x = read_txt("./data/附件5：动态轨迹数据.txt")
    test_x = test_x[:, :, 2]
    #show_original_data(test_x)
    test_x =test_x.T
    xhat = []
    for i in range(4):
        # raw_data = device_data[i]
        raw_data = test_x[i]
        xhat.append(KalmanFilter(raw_data, n_iter=len(raw_data)))
    xhat = np.array(xhat)[:,1::]
    test_x =xhat.T
    classification_model=joblib.load("task4_data_classification_rfc.model")
    normal_data_regression_model = joblib.load("task2_normal_dataset_regression_MLP.model")
    abnormal_data_regression_model = joblib.load("task2_abnormal_dataset_regression_MLP.model")
    global d0, d1, d2, d3
    trajectory =[]
    for i in range(test_x.shape[0]):
        d0,d1,d2,d3 = test_x[i, :]
        tag = np.array(calculate_4_tag_position())
        classification_result=classification_model.predict(test_x[i, :].reshape((1,-1)))
        if classification_result[0]==1: # 产生1即为无干扰的程序
            tag = normal_data_regression_model.predict(tag.flatten().reshape(1,-1))  # 传感器数据融合
        elif classification_result[0]==0:
            tag = abnormal_data_regression_model.predict(tag.flatten().reshape(1,-1)) # 传感器数据融合
        else:
            raise ValueError("预测的标签有误")
        trajectory.append(tag)
    trajectory=np.array(trajectory).reshape((len(trajectory),3))
    return trajectory

def window_abnormal_data_detection(trajectory):
    iteration_num=2
    thre=2 #2σ
    window_length = 8 # 滑动窗口的长度
    for _ in range(iteration_num):
        for i in range(trajectory.shape[0]-window_length):
            window_data = trajectory[i:i+window_length,:]
            window_data_mean = window_data.mean(axis=0)
            window_data_std  = window_data.std(axis=0)
            low_thre = window_data_mean - window_data_std * thre  # 去除离群点
            high_thre = window_data_mean + window_data_std * thre  # 去除离群点
            for j in range(window_data.shape[1]):
                for k in range(window_data.shape[0]):
                    if window_data[k,j]>high_thre[j] or window_data[k,j]<low_thre[j]:
                        window_data[k,j]=window_data_mean[j]
                        trajectory[i,j]=window_data_mean[j]
                        window_data_mean = window_data.mean(axis=0)
                        window_data_std = window_data.std(axis=0)
                        low_thre = window_data_mean - window_data_std * thre  # 更新阈值
                        high_thre = window_data_mean + window_data_std * thre  # 更新阈值
    return trajectory

if __name__ == '__main__':
    trajectory=get_trajectory()
    #trajectory=window_abnormal_data_detection(trajectory)  # 滑动窗口去除离群点
    '''
    #卡尔曼具有滞后性，保险起见，不加卡尔曼了
    trajectory=trajectory.T
    xhat = []
    # 卡尔曼滤波进一步保证数据准确性
    for i in range(3):
        # raw_data = device_data[i]
        raw_data = trajectory[i]
        xhat.append(KalmanFilter(raw_data, n_iter=len(raw_data)))
    xhat = np.array(xhat)[:,1::]
    trajectory =xhat.T
    '''
    show_3D_figure(trajectory)
    breakpoint()