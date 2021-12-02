import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class basic_propress():
    def __init__(self,training_df,testing_df,col_name,name):
        self.training_df = training_df
        self.testing_df = testing_df
        self.col_name = col_name
        self.name = name
    # def get_target_label_idx(labels, targets):
    #     """
    #     Get the indices of labels that are included in targets.
    #     :param labels: array of labels
    #     :param targets: list/tuple of target labels
    #     :return: list with indices of target labels
    #     """
    #     return np.argwhere(~np.isin(labels, targets)).flatten().tolist()
    #
    #
    # def global_contrast_normalization(x: torch.tensor, scale='l2'):
    #     """
    #     Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    #     which is either the standard deviation, L1- or L2-norm across features (pixels).
    #     Note this is a *per sample* normalization globally across features (and not across the dataset).
    #     """
    #
    #     assert scale in ('l1', 'l2')
    #
    #     n_features = int(np.prod(x.shape))
    #
    #     mean = torch.mean(x)  # mean over all features (pixels) per sample
    #     x -= mean
    #
    #     if scale == 'l1':
    #         x_scale = torch.mean(torch.abs(x))
    #
    #     if scale == 'l2':
    #         x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    #
    #     x /= x_scale
    #
    #     return x

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, item):
        pass



def minmax_scale_values(training_df,testing_df, col_name):
    scaler = MinMaxScaler()
    # scaler = scaler.fit(training_df[col_name].reshape(-1, 1))
    train_values_standardized = scaler.fit_transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized


def encode_text(training_df, testing_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns:
            testing_df[dummy_name] = testing_set_dummies[x]
        else:
            testing_df[dummy_name] = np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)

def label_attract(row,df_normal,df_malware):
    if (row["label"] == 1):
        return df_normal.append[row]
    else:
        return df_malware.append[row]

# def decide_normal(df,df_normal):
#     for label in range(df):
#         if (df["label"]==1):
#             df_normal.append[]

def process_tsne_unsw(num):
    training_df = pd.read_csv("./unsw/UNSW_NB15_training-set.csv")
    testing_df = pd.read_csv("./unsw/UNSW_NB15_testing-set.csv")
    training_df = pd.DataFrame(training_df)
    testing_df = pd.DataFrame(testing_df)
    df = pd.concat([training_df, testing_df])
    # print(df)
    df = df.drop("attack_cat", axis=1)
    training_df = training_df.drop("attack_cat", axis=1)
    testing_df = testing_df.drop("attack_cat", axis=1)

    sympolic_columns = ["proto", "service", "state"]
    label_column = "Class"
    for column in df.columns:
        if column in sympolic_columns:
            encode_text(training_df, testing_df, column)
        elif not column == label_column:
            minmax_scale_values(training_df, testing_df, column)

    # 处理了第一个文件夹training
    X_train_malware = training_df[training_df['label'] != 0]
    X_train_normal = training_df[training_df['label'] == 0]
    X_train_normal = X_train_normal.drop("label",axis=1)
    X_train_malware = X_train_malware.drop("label",axis=1)
    # 处理第二个文件夹testing 1为恶意
    X_test_malware = testing_df[testing_df['label'] != 0]
    X_test_normal = testing_df[testing_df['label'] == 0]

    X_train_normal = X_train_normal.drop("id",axis=1)
    X_test_normal = X_test_normal.drop("id",axis=1)
    X_test_normal = X_test_normal.drop("label",axis=1)
    X_test_malware = X_test_malware.drop("id",axis=1)
    X_test_malware = X_test_malware.drop("label",axis=1)
    normal = np.vstack((X_train_normal,X_test_normal))

    X_train_malware = X_train_malware.drop("id",axis=1)
    malware = np.vstack((X_train_malware,X_test_malware))
    np.random.shuffle(malware)
    np.random.shuffle(normal)

    num2 = int(num/10)
    normal_test = normal[:num]
    y_normal = np.zeros((len(normal_test),1))
    malware_test = malware[:num2]
    y_malware = np.ones((len(malware_test),1))
    x_test = np.vstack((normal_test,malware_test))
    y_test = np.vstack((y_normal,y_malware))

    return x_test,y_test

def main_process_unsw(i):
    training_df = pd.read_csv("./unsw/UNSW_NB15_training-set.csv")
    testing_df = pd.read_csv("./unsw/UNSW_NB15_testing-set.csv")
    training_df = pd.DataFrame(training_df)
    testing_df = pd.DataFrame(testing_df)
    # print(training_df)
    # 把数据集合在一起，目前还没有用过
    df = pd.concat([training_df, testing_df])
    # print(df)
    df = df.drop("attack_cat", axis=1)

    training_df = training_df.drop("attack_cat", axis=1)
    testing_df = testing_df.drop("attack_cat", axis=1)

    sympolic_columns = ["proto", "service", "state"]
    label_column = "Class"
    for column in df.columns:
        if column in sympolic_columns:
            encode_text(training_df, testing_df, column)
        elif not column == label_column:
            minmax_scale_values(training_df, testing_df, column)

    # 处理了第一个文件夹training
    X_train_malware = training_df[training_df['label'] != 0]
    X_train_normal = training_df[training_df['label'] == 0]
    X_train_normal = X_train_normal.drop("label",axis=1)
    X_train_malware = X_train_malware.drop("label",axis=1)
    # 处理第二个文件夹testing 1为恶意
    X_test_malware = testing_df[testing_df['label'] != 0]
    X_test_normal = testing_df[testing_df['label'] == 0]

    y_test_malware = np.ones((len(X_test_malware),1))
    y_test_normal = np.zeros((len(X_test_normal),1))
    #X_test = pd.concat([X_test_normal, X_test_malware])
    #X_test = X_test.drop("label",axis=1)
    X_train_normal = X_train_normal.drop("id",axis=1)
    X_test_normal = X_test_normal.drop("id",axis=1)
    X_test_normal = X_test_normal.drop("label",axis=1)
    X_test_malware = X_test_malware.drop("id",axis=1)
    X_test_malware = X_test_malware.drop("label",axis=1)
    normal = np.vstack((X_train_normal,X_test_normal))
    print('正常流量：', normal.shape)

    X_train_malware = X_train_malware.drop("id",axis=1)
    malware = np.vstack((X_train_malware,X_test_malware))
    print('恶意流量：', malware.shape)
    np.random.shuffle(malware)

    test_malware = malware[:700]
    y_test_malware = np.ones((len(test_malware), 1))
    test_normal = normal[:i*700]
    y_test_normal = np.zeros((len(test_normal), 1))
    train_normal = normal[i*700:]
    x_test = np.vstack((test_normal,test_malware))
    y_test = np.vstack((y_test_normal,y_test_malware))
    #X_test = X_test.drop("id",axis=1)
    print('x_test:',x_test.shape)
    #print('y_test:',y_test.shape)
    print('训练集中的正常流量：',train_normal.shape)
    #print('训练集中的恶意流量：',X_train_malware.shape)

    #print('测试集中的流量形状：',X_test.shape)
    return train_normal,x_test,y_test

def get_typeofmalware(malware_name):
    """
    要将测试集中不同种类的恶意流量进行分离，并且标准化和归一化处理
    :return:
    """
    training_df = pd.read_csv("./unsw/UNSW_NB15_training-set.csv")
    testing_df = pd.read_csv("./unsw/UNSW_NB15_testing-set.csv")
    training_df = pd.DataFrame(training_df)
    testing_df = pd.DataFrame(testing_df)
    df = pd.concat([training_df, testing_df])
    df = df.drop("attack_cat", axis=1)
    training_df = training_df.drop("attack_cat", axis=1)
    # training_df = training_df.drop("label", axis=1)
    # print(training_df.shape)

    x_test_analysis = testing_df[testing_df['attack_cat']==malware_name]
    x_test_analysis = x_test_analysis.drop("attack_cat",axis = 1)
    # print(x_test_analysis.shape)
    sympolic_columns = ["proto", "service", "state"]
    label_column = "Class"
    for column in df.columns:
        if column in sympolic_columns:
            encode_text(training_df, x_test_analysis, column)
        elif not column == label_column:
            minmax_scale_values(training_df, x_test_analysis, column)

    x_test_analysis = x_test_analysis.drop("label", axis=1)
    x_test_analysis = x_test_analysis.drop("id", axis=1)
    print('x_test_malware:',x_test_analysis.shape)
    # print(x_test_analysis.head)
    # print('training_df:',training_df.shape)
    return x_test_analysis

if __name__ == '__main__':
    # main_process(10)
    x_test_analysis = get_typeofmalware('Analysis')
    #print(x_test_analysis.head)