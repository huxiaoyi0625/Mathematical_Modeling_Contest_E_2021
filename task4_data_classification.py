from task1_plotFourPictures import read_txt
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import joblib
from sklearn.neural_network import MLPClassifier

def read_dataset(kinds=None, spilt_normal_data_and_abnormal_data=False):
    if kinds is None:
        kinds = ["正常", "异常"]
    data = []
    num_of_kind_data = []
    for kind in kinds:
        #kind_data = np.array(pd.read_csv(f"remove_outliner_data/{kind}数据/{1}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]
        kind_data = np.array(pd.read_csv(f"cleaned_data/{kind}数据/{1}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]
        for i in range(2, 325):
            #kind_data = np.row_stack((kind_data, np.array(
                #pd.read_csv(f"remove_outliner_data/{kind}数据/{i}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]))
            kind_data = np.row_stack((kind_data, np.array(
                pd.read_csv(f"cleaned_data/{kind}数据/{i}.{kind}.csv", usecols=[1, 2, 3, 4]))[1::, :]))
        num_of_kind_data.append(kind_data.shape[0])
        data.append(kind_data)
    if spilt_normal_data_and_abnormal_data:
        normal_data = np.column_stack((np.array(data[0]), np.ones(num_of_kind_data[0])))
        abnormal_data = np.column_stack((np.array(data[1]), np.zeros(num_of_kind_data[1])))
        return (normal_data, abnormal_data)
    else:
        data = np.concatenate(data, axis=0)
        label = np.concatenate((np.ones(num_of_kind_data[0]), np.zeros(num_of_kind_data[1])), axis=0)  # 1是正常数据,0是异常数据
        data = np.column_stack((data, label))
        np.random.shuffle(data)  # 数据集随机打散
        return data


def train_random_forest(data):
    # 建立随机森林

    X = data[:, 0:4]
    Y = data[:, 4]
    train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=0, test_size=0.2)

    rfc = RandomForestClassifier(n_estimators=20, random_state=90)
    rfc.fit(train_X, train_y)
    val_predictions = rfc.predict(val_X)
    print(classification_report(val_y, val_predictions))
    # 用交叉验证计算得分
    score_pre = cross_val_score(rfc, X, Y, cv=3).mean()
    print(f"交叉验证结果为:{score_pre}")
    return rfc


def get_dissimilar_data_mean_std(file_path):
    data = pd.read_csv(file_path, usecols=[1], encoding="gbk")
    the_number_of_data = np.array(data).reshape(-1)
    return the_number_of_data.mean(), the_number_of_data.std()

def save_forest_model():
    data = read_dataset()
    model = train_random_forest(data)
    joblib.dump(model,"task4_data_classification_rfc.model")

def training_MLP_Model():
    data = read_dataset()
    X = data[:, 0:4]
    Y = data[:, 4]
    train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=0, test_size=0.2)

    MLP = MLPClassifier(solver='adam',
                        hidden_layer_sizes=(50,30,10), random_state=1)
    MLP.fit(train_X, train_y)
    val_predictions = MLP.predict(val_X)
    print(classification_report(val_y, val_predictions))
    joblib.dump(MLP, "task4_data_classification_MLP.model")
    return MLP

if __name__ == '__main__':
    #training_MLP_Model()
    save_forest_model()
    test_x = read_txt("./data/附件4：测试集.txt")
    test_x = test_x[:, :, 2]
    model = joblib.load("task4_data_classification_rfc.model")
    test_y = model.predict(test_x)
    with open("./submit/task4/对十组信号的分类.csv", "w+", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
        for i in range(test_y.shape[0]):
            if test_y[i] == 0:
                csvwriter.writerow(["test_data_" + str(i + 1), "数据异常(有信号干扰)"])
            if test_y[i] == 1:
                csvwriter.writerow(["test_data_" + str(i + 1), "数据正常(无信号干扰)"])
    breakpoint()
