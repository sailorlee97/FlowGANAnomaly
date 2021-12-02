"""
@Time    : 2021/7/14 15:06
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: process_cicids.py
@Software: PyCharm
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os



def test_bot():
    test_bot1 = pd.read_csv('../cicids2017/pcap_raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
    bot1 = test_bot1[test_bot1[' Label'] == 'Web Attack XSS']
    bot1.to_csv('../cicids2017/Web Attack XSS.csv',mode='a', header=False)

def get_malware(name):
    for info in os.listdir('../cicids2017/pcap_raw'):
        domain = os.path.abspath(r'../cicids2017/pcap_raw')  # 获取文件夹的路径
        info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
        data = pd.read_csv(info)
        bot1 = data[data[' Label']== name]
        bot1.to_csv('../cicids2017/test_%s.csv'%name, mode='a', header=False)

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

def process_tsne_data():
    training_df = pd.read_csv("./cicids2017/benign.csv",header=None)
    testing_df1 = pd.read_csv("./cicids2017/bot.csv",header=None)
    testing_df2 = pd.read_csv("./cicids2017/test_DDoS.csv", header=None)
    testing_df3 = pd.read_csv("./cicids2017/test_DoS GoldenEye.csv", header=None)
    testing_df4 = pd.read_csv("./cicids2017/test_DoS Hulk.csv", header=None)
    testing_df5 = pd.read_csv("./cicids2017/test_DoS Slowhttptest.csv", header=None)
    testing_df6 = pd.read_csv("./cicids2017/test_FTP-Patator.csv", header=None)
    testing_df7 = pd.read_csv("./cicids2017/test_Heartbleed.csv", header=None)
    testing_df8 = pd.read_csv("./cicids2017/test_DoS slowloris.csv", header=None)
    testing_df9 = pd.read_csv("./cicids2017/test_Infiltration.csv", header=None)
    testing_df10 = pd.read_csv("./cicids2017/test_PortScan.csv", header=None)
    testing_df11 = pd.read_csv("./cicids2017/test_SSH-Patator.csv", header=None)
    testing_df12 = pd.read_csv("./cicids2017/Web Attack Brute Force.csv", header=None)
    testing_df13 = pd.read_csv("./cicids2017/Web Attack Sql Injection.csv", header=None)
    testing_df14 = pd.read_csv("./cicids2017/Web Attack XSS.csv", header=None)

    training_normal_df = training_df
    training_normal_df = training_normal_df.iloc[:, :-1]
    training_normal_df = training_normal_df.drop(0, axis=1)
    #print(training_normal_df.head)
    training_normal_df = training_normal_df.values
    #print(training_normal_df)
    training_normal_df = np.delete(training_normal_df,np.where(np.isinf(training_normal_df))[0],axis=0)

    malware = pd.concat([testing_df1, testing_df2, testing_df3, testing_df4, testing_df5, testing_df6, testing_df7,
                         testing_df8, testing_df9, testing_df10, testing_df11, testing_df12, testing_df13,
                         testing_df14])
    malware = malware.iloc[:, :-1]
    malware = malware.drop(0, axis=1)
    malware = malware.values
    malware = np.delete(malware, np.where(np.isinf(malware))[0], axis=0)
    np.random.shuffle(malware)

    test_malware = malware[:8000]
    test_normal = training_normal_df[:8000]
    y_test_normal = np.zeros((len(test_normal), 1))
    y_test_malware  = np.ones((len(test_malware), 1))
    x_test = np.vstack((test_normal,test_malware))
    y_test = np.vstack((y_test_normal, y_test_malware))

    return x_test,y_test

# ===========================================================================
# -------------------------训练集和测试集分开获取------------------------------
# ===========================================================================

def process_cicids_train(i):
    training_df = pd.read_csv("./cicids2017/test_BENIGN.csv", header=None)
    training_normal_df = training_df
    training_normal_df = training_normal_df.iloc[:, :-1]
    training_normal_df = training_normal_df.drop(0, axis=1)
    # print(training_normal_df.head)
    training_normal_df = training_normal_df.values
    # print(training_normal_df)
    training_normal_df = np.delete(training_normal_df, np.where(np.isinf(training_normal_df))[0], axis=0)
    # print(training_normal_df.shape)
    train_normal = training_normal_df[6000 * i:]
    #归一化处理
    scaler = MinMaxScaler()
    train_normal = scaler.fit_transform(train_normal)
    y_train = np.zeros((len(train_normal), 1))
    return train_normal,y_train

def process_cicids_test(i):
    training_df = pd.read_csv("./cicids2017/test_BENIGN.csv", header=None)

    testing_df1 = pd.read_csv("./cicids2017/bot.csv", header=None)
    testing_df2 = pd.read_csv("./cicids2017/test_DDoS.csv", header=None)
    testing_df3 = pd.read_csv("./cicids2017/test_DoS GoldenEye.csv", header=None)
    testing_df4 = pd.read_csv("./cicids2017/test_DoS Hulk.csv", header=None)
    testing_df5 = pd.read_csv("./cicids2017/test_DoS Slowhttptest.csv", header=None)
    testing_df6 = pd.read_csv("./cicids2017/test_FTP-Patator.csv", header=None)
    testing_df7 = pd.read_csv("./cicids2017/test_Heartbleed.csv", header=None)
    testing_df8 = pd.read_csv("./cicids2017/test_DoS slowloris.csv", header=None)
    testing_df9 = pd.read_csv("./cicids2017/test_Infiltration.csv", header=None)
    testing_df10 = pd.read_csv("./cicids2017/test_PortScan.csv", header=None)
    testing_df11 = pd.read_csv("./cicids2017/test_SSH-Patator.csv", header=None)
    testing_df12 = pd.read_csv("./cicids2017/Web Attack Brute Force.csv", header=None)
    testing_df13 = pd.read_csv("./cicids2017/Web Attack Sql Injection.csv", header=None)
    testing_df14 = pd.read_csv("./cicids2017/Web Attack XSS.csv", header=None)

    training_normal_df = training_df
    training_normal_df = training_normal_df.iloc[:, :-1]
    training_normal_df = training_normal_df.drop(0, axis=1)
    # print(training_normal_df.head)
    training_normal_df = training_normal_df.values
    # print(training_normal_df)
    training_normal_df = np.delete(training_normal_df, np.where(np.isinf(training_normal_df))[0], axis=0)
    # print(training_normal_df.shape)

    malware = pd.concat([testing_df1, testing_df2, testing_df3, testing_df4, testing_df5, testing_df6, testing_df7,
                         testing_df8, testing_df9, testing_df10, testing_df11, testing_df12, testing_df13,
                         testing_df14])
    malware = malware.iloc[:, :-1]
    malware = malware.drop(0, axis=1)
    malware = malware.values
    malware = np.delete(malware, np.where(np.isinf(malware))[0], axis=0)
    np.random.shuffle(malware)
    test_malware = malware[:6000]
    test_normal = training_normal_df[:6000 * i]

    y_test_normal = np.zeros((len(test_normal), 1))
    y_test_malware = np.ones((len(test_malware), 1))
    x_test = np.vstack((test_normal, test_malware))
    y_test = np.vstack((y_test_normal, y_test_malware))
    print('y_test:', y_test.shape)
    print('X_test:',x_test.shape)
    #归一化处理
    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)
    return x_test,y_test

# ===========================================================================
# -------------------------训练集和测试集同时获取------------------------------
# ===========================================================================

def main_process_cicids(i):
    # for info in os.listdir('../cicids2017'):
    #     domain = os.path.abspath(r'../cicids2017')  # 获取文件夹的路径
    #     info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
    #     data = pd.read_csv(info)cicids2017/test_BENIGN.csv

    training_df = pd.read_csv("./cicids2017/test_BENIGN.csv",header=None)

    testing_df1 = pd.read_csv("./cicids2017/bot.csv",header=None)
    testing_df2 = pd.read_csv("./cicids2017/test_DDoS.csv", header=None)
    testing_df3 = pd.read_csv("./cicids2017/test_DoS GoldenEye.csv", header=None)
    testing_df4 = pd.read_csv("./cicids2017/test_DoS Hulk.csv", header=None)
    testing_df5 = pd.read_csv("./cicids2017/test_DoS Slowhttptest.csv", header=None)
    testing_df6 = pd.read_csv("./cicids2017/test_FTP-Patator.csv", header=None)
    testing_df7 = pd.read_csv("./cicids2017/test_Heartbleed.csv", header=None)
    testing_df8 = pd.read_csv("./cicids2017/test_DoS slowloris.csv", header=None)
    testing_df9 = pd.read_csv("./cicids2017/test_Infiltration.csv", header=None)
    testing_df10 = pd.read_csv("./cicids2017/test_PortScan.csv", header=None)
    testing_df11 = pd.read_csv("./cicids2017/test_SSH-Patator.csv", header=None)
    testing_df12 = pd.read_csv("./cicids2017/Web Attack Brute Force.csv", header=None)
    testing_df13 = pd.read_csv("./cicids2017/Web Attack Sql Injection.csv", header=None)
    testing_df14 = pd.read_csv("./cicids2017/Web Attack XSS.csv", header=None)


    training_normal_df = training_df
    training_normal_df = training_normal_df.iloc[:, :-1]
    training_normal_df = training_normal_df.drop(0, axis=1)
    #print(training_normal_df.head)
    training_normal_df = training_normal_df.values
    #print(training_normal_df)
    training_normal_df = np.delete(training_normal_df,np.where(np.isinf(training_normal_df))[0],axis=0)
    print(training_normal_df.shape)


    #test_normal_df = training_df.iloc[1000001:1048570]
    #test_normal_df_ = test_normal_df.iloc[:, :-1]
    #test_normal_df_ = test_normal_df_.drop(0, axis=1)
    #test_normal_df_ = test_normal_df_.values
    #test_normal_df_ = np.delete(test_normal_df_,np.where(np.isinf(test_normal_df_))[0],axis=0)
    #y_test_normal = np.zeros((len(test_normal_df_), 1))
    #X_test = pd.concat([test_normal_df, testing_df1,testing_df2,testing_df3,testing_df4,testing_df5,testing_df6,testing_df7,
                        #testing_df8,testing_df9,testing_df10,testing_df11,testing_df12,testing_df13,testing_df14])
    malware = pd.concat([testing_df1,testing_df2,testing_df3,testing_df4,testing_df5,testing_df6,testing_df7,
                        testing_df8,testing_df9,testing_df10,testing_df11,testing_df12,testing_df13,testing_df14])
    malware = malware.iloc[:, :-1]
    malware = malware.drop(0, axis=1)
    malware = malware.values
    malware = np.delete(malware,np.where(np.isinf(malware))[0],axis=0)
    print(malware.shape)
    np.random.shuffle(malware)
    test_malware = malware[:6000]
    test_normal = training_normal_df[:6000*i]
    train_normal = training_normal_df[6000*i:]
    y_test_normal = np.zeros((len(test_normal), 1))
    y_test_malware  = np.ones((len(test_malware), 1))
    x_test = np.vstack((test_normal,test_malware))
    y_test = np.vstack((y_test_normal, y_test_malware))
    print('y_test:',y_test.shape)
    # X_test.to_csv('../csv_data/x_test_cicids.csv')
    # training_normal_df.to_csv('../csv_data/training_normal_cicids.csv')
    #training_normal_df = training_normal_df.iloc[:, :-1]

    #print(training_normal_df.head)

    #round(X_test,2)
    #round(training_normal_df,2)
    # training_normal_df.to_csv('../csv_data/training_normal_cicids.csv')
    # X_test.to_csv('../csv_data/X_test_cicids.csv')
    # print('x test',X_test.head)
    # print(training_normal_df.isnull().any())
    # print(training_normal_df.describe())
    # print(X_test.isnull().any())
    # print(X_test.describe())

    #print(training_normal_df.head)
    #X_test = X_test.iloc[:, :-1]
    #X_test = X_test.drop(0, axis=1)
    #print(X_test.head)
    #X_test = X_test.values
    #print(X_test)
    #X_test = np.delete(X_test, np.where(np.isinf(X_test))[0], axis=0)
    print('X_test:',x_test.shape)
    #归一化处理
    scaler = MinMaxScaler()
    train_normal = scaler.fit_transform(train_normal)
    x_test = scaler.fit_transform(x_test)
    #print('training_normal_df:',training_normal_df.shape)
    #print(training_normal_df)

    #print('X_test:',X_test.shape)
    #print(X_test)
    # print('x test', X_test.head)
    # for column in training_normal_df.columns:
    #     minmax_scale_values(training_normal_df,X_test,column)

    #training_normal_df.to_csv('../csv_data/training_normal_cicids.csv')
    #X_test.to_csv('../csv_data/X_test_cicids.csv')
    return train_normal,x_test,y_test,len(test_normal),len(test_malware)

# ===========================================================================
# -------------------------获取不同种类的恶意流量------------------------------
# 下面是将dos攻击和web攻击进行合并，因此bot、dos和web三类攻击需要单独提取，其他恶意
# 流量直接按照名字提取即可。
# ===========================================================================

def get_bot_malware():
    training_df = pd.read_csv("./cicids2017/bot.csv",header=None)
    training_malware_df = training_df
    training_malware_df = training_malware_df.iloc[:, :-1]
    training_malware_df = training_malware_df.drop(0, axis=1)
    # print(training_normal_df.head)
    training_malware_df = training_malware_df.values
    # print(training_normal_df)
    training_malware_df = np.delete(training_malware_df, np.where(np.isinf(training_malware_df))[0], axis=0)

    scaler = MinMaxScaler()
    training_malware_df = scaler.fit_transform(training_malware_df)
    y_malware = np.ones((len(training_malware_df), 1))

    return training_malware_df,y_malware


def get_type_malware(name):
    """
    :param name: DDoS FTP-Patator Heartbleed Infiltration PortScan SSH-Patator
    :return: malware
    """
    df = pd.read_csv("./cicids2017/test_%s.csv"%name, header=None)
    training_malware_df = df
    training_malware_df = training_malware_df.iloc[:, :-1]
    training_malware_df = training_malware_df.drop(0, axis=1)
    # print(training_normal_df.head)
    training_malware_df = training_malware_df.values
    # print(training_normal_df)
    training_malware_df = np.delete(training_malware_df, np.where(np.isinf(training_malware_df))[0], axis=0)

    scaler = MinMaxScaler()
    training_malware_df = scaler.fit_transform(training_malware_df)
    y_malware = np.ones((len(training_malware_df), 1))

    return training_malware_df, y_malware

def get_dos_malware():
    testing_df3 = pd.read_csv("./cicids2017/test_DoS GoldenEye.csv", header=None)
    testing_df4 = pd.read_csv("./cicids2017/test_DoS Hulk.csv", header=None)
    testing_df5 = pd.read_csv("./cicids2017/test_DoS Slowhttptest.csv", header=None)
    testing_df8 = pd.read_csv("./cicids2017/test_DoS slowloris.csv", header=None)
    dos = pd.concat([testing_df3, testing_df4, testing_df5,testing_df8])
    dos = dos.iloc[:, :-1]
    dos = dos.drop(0, axis=1)
    dos = dos.values
    malware = np.delete(dos, np.where(np.isinf(dos))[0], axis=0)
    np.random.shuffle(malware)

    scaler = MinMaxScaler()
    training_malware_df = scaler.fit_transform(malware)
    y_malware = np.ones((len(training_malware_df), 1))
    return training_malware_df,y_malware


def get_web_malware():
    testing_df12 = pd.read_csv("./cicids2017/Web Attack Brute Force.csv", header=None)
    testing_df13 = pd.read_csv("./cicids2017/Web Attack Sql Injection.csv", header=None)
    testing_df14 = pd.read_csv("./cicids2017/Web Attack XSS.csv", header=None)
    web_attack = pd.concat([testing_df12, testing_df13, testing_df14])
    web_attack = web_attack.iloc[:, :-1]
    web_attack = web_attack.drop(0, axis=1)
    web_attack = web_attack.values
    malware = np.delete(web_attack, np.where(np.isinf(web_attack))[0], axis=0)
    np.random.shuffle(malware)

    scaler = MinMaxScaler()
    training_malware_df = scaler.fit_transform(malware)
    y_malware = np.ones((len(training_malware_df), 1))
    return training_malware_df, y_malware

if __name__ == '__main__':
    #
    # 获取不同分类恶意数据集进行
    #

    # list_name = ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'FTP-Patator', 'Heartbleed', 'DoS slowloris', 'Infiltration',
    #              'PortScan','SSH-Patator','Web Attack Brute Force','Web Attack Sql Injection','Web Attack XSS']
    # for item in list_name:
    #     get_malware(item)
    # test_bot()
    main_process_cicids(1)