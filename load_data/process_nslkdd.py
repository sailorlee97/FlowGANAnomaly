"""
@Time    : 2021/7/14 9:15
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: process_nslkdd.py
@Software: PyCharm
"""
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'difficulty'
]

dos_attacks=["back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpgetattack","snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps","httptunnel"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]
classes=["Normal","Dos","R2L","U2R","Probe"]

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

def label_attack (row,classes):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]


def main_process_nsl(i):
    training_df = pd.read_csv("./nsk-kdd/KDDTest+.csv")
    testing_df = pd.read_csv("./nsk-kdd/KDDTrain+.csv")

    training_df = pd.DataFrame(training_df)
    testing_df = pd.DataFrame(testing_df)
    df = pd.concat([training_df, testing_df])
    # print(df)
    df = df.drop("number", axis=1)
    df = df.drop("num", axis=1)
    df = df.drop("label", axis=1)

    # print(df)
    # print(training_df.columns)
    training_df = training_df.drop("number", axis=1)
    testing_df = testing_df.drop("number", axis=1)
    sympolic_columns = ["protocol_type", "service", "flag"]

    for column in df.columns:
        if column in sympolic_columns:
            encode_text(training_df, testing_df, column)
        else:
            minmax_scale_values(training_df, testing_df, column)

    # 处理了第一个文件夹training
    X_train_malware = training_df[training_df['label'] != 'normal']
    X_train_normal = training_df[training_df['label'] == 'normal']
    # print(X_train_normal.columns)
    # print(X_train_normal.shape)
    X_train_normal = X_train_normal.drop("label", axis=1)
    X_train_malware = X_train_malware.drop("label", axis=1)
    X_train_normal = X_train_normal.drop("num",axis=1)
    X_train_malware = X_train_malware.drop("num",axis=1)

    # 处理第二个文件夹testing 1为恶意

    X_test_normal = testing_df[testing_df['label'] == 'normal']
    X_test_normal = X_test_normal.drop("label", axis=1)
    X_test_normal = X_test_normal.drop("num", axis=1)

    X_test_malware = testing_df[testing_df['label'] != 'normal']
    #X_test = pd.concat([X_test_normal, X_test_malware])
    X_test_malware = X_test_malware.drop("label", axis=1)
    X_test_malware = X_test_malware.drop("num", axis=1)



    #整合一起
    normal = np.vstack((X_train_normal,X_test_normal))
    malware = np.vstack((X_train_malware,X_test_malware))
    np.random.shuffle(malware)

    print('normal:',normal.shape)
    print('malware:',malware.shape)

    test_malware = malware[:500]
    y_test_malware = np.ones((len(test_malware), 1))
    test_normal = normal[:i*500]
    y_test_normal = np.zeros((len(test_normal), 1))
    train_normal = normal[i*500:]
    x_test = np.vstack((test_normal,test_malware))
    y_test = np.vstack((y_test_normal,y_test_malware))
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    return train_normal, x_test, y_test

def process_tsne_nsl(num):
    training_df = pd.read_csv("./nsk-kdd/KDDTest+.csv")
    testing_df = pd.read_csv("./nsk-kdd/KDDTrain+.csv")
    training_df = pd.DataFrame(training_df)
    testing_df = pd.DataFrame(testing_df)
    df = pd.concat([training_df, testing_df])
    # print(df)
    df = df.drop("number", axis=1)
    df = df.drop("num", axis=1)
    df = df.drop("label", axis=1)

    # print(df)
    # print(training_df.columns)
    training_df = training_df.drop("number", axis=1)
    testing_df = testing_df.drop("number", axis=1)
    sympolic_columns = ["protocol_type", "service", "flag"]

    for column in df.columns:
        if column in sympolic_columns:
            encode_text(training_df, testing_df, column)
        else:
            minmax_scale_values(training_df, testing_df, column)

    # 处理了第一个文件夹training
    X_train_malware = training_df[training_df['label'] != 'normal']
    X_train_normal = training_df[training_df['label'] == 'normal']
    # print(X_train_normal.columns)
    # print(X_train_normal.shape)
    X_train_normal = X_train_normal.drop("label", axis=1)
    X_train_malware = X_train_malware.drop("label", axis=1)
    X_train_normal = X_train_normal.drop("num",axis=1)
    X_train_malware = X_train_malware.drop("num",axis=1)

    # 处理第二个文件夹testing 1为恶意

    X_test_normal = testing_df[testing_df['label'] == 'normal']
    X_test_normal = X_test_normal.drop("label", axis=1)
    X_test_normal = X_test_normal.drop("num", axis=1)

    X_test_malware = testing_df[testing_df['label'] != 'normal']
    #X_test = pd.concat([X_test_normal, X_test_malware])
    X_test_malware = X_test_malware.drop("label", axis=1)
    X_test_malware = X_test_malware.drop("num", axis=1)

    #整合一起
    normal = np.vstack((X_train_normal,X_test_normal))
    malware = np.vstack((X_train_malware,X_test_malware))
    np.random.shuffle(malware)
    np.random.shuffle(normal)

    x_normal = normal[:num]
    y__normal = np.zeros((len(x_normal), 1))
    num_2 = int(num/10)
    x_malware = malware[:num_2]
    y__malware = np.ones((len(x_malware), 1))

    x_test = np.vstack((x_normal,x_malware))
    y_test = np.vstack((y__normal,y__malware))

    return x_test,y_test

def get_typeofmalware(malware_name):
    training_df = pd.read_csv("../nsk-kdd/KDDTest+.csv")
    testing_df = pd.read_csv("../nsk-kdd/KDDTrain+.csv")
    df = pd.concat([training_df, testing_df])
    df["Class"] = df.apply(label_attack, axis=1)

if __name__ == '__main__':
    main_process_nsl (10)