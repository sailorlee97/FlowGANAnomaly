"""
@Time    : 2022/8/14 20:16
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: cicdos2019.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def _drop_label(df,list):
    df = df.drop(df.columns[[0]], axis=1)
    for i in list:
        df = df.drop(i, axis=1)
    return df

def _scaler_normal(num):
    scaler = MinMaxScaler()
    new_num = scaler.fit_transform(num.T)

    return new_num.T

def _check_array_deleteinfnan(arr):

    arr = arr[:, ~np.isnan(arr).any(axis=0)]
    arr = arr[:, ~np.isinf(arr).any(axis=0)]

    return arr

def _findAllFile(orginal):
    list = []
    for root, ds, fs in os.walk(orginal):
        for f in fs:
            list.append(orginal+"/"+f)
    return list

def _get_normal_csv(f):
    df = pd.read_csv(f)
    df = df[df[' Label'] == 'BENIGN']
    return df

def combine_csv(f):

    all_filenames = _findAllFile(f)

    combined_csv = pd.concat([_get_normal_csv(f) for f in all_filenames ])

    return combined_csv


def get_flow_minmaxscalr(num):
    return _scaler_normal(num)

def fig_num():
    df1 = pd.read_csv('./cicdos/UDP.csv')
    print(df1)

def process_dos():

    list = ['Flow ID',' Source IP',' Destination IP',' Timestamp','SimillarHTTP',' Label']
    df_normal = combine_csv('./cicdos')
    df1 = pd.read_csv('./cicdos/Portmap.csv')

    #df_normal = df[df[' Label']=='BENIGN']
    df_malware = df1[df1[' Label']=='Portmap']

    df_normal = _drop_label(df_normal,list)
    df_normal.to_csv("dos_normal")
    df_normal = df_normal.values
    df_normal = _check_array_deleteinfnan(df_normal)
    df_normal = _scaler_normal(df_normal)
    #df_normal.to_csv("dos_normal")

    print(df_normal.shape)
    df_malware = _drop_label(df_malware,list)
    #df_malware.to_csv("dos_malware")
    df_malware = df_malware.values
    df_malware = _check_array_deleteinfnan(df_malware)
    df_malware = _scaler_normal(df_malware)
    train_normal = df_normal[:27000]
    test_normal = df_normal[27000:]
    y_test_normal = np.zeros((len(test_normal), 1))
    test_malware = df_malware[:600]
    y_test_malware = np.ones((len(test_malware), 1))
    x_test = np.vstack((test_normal, test_malware))
    y_test = np.vstack((y_test_normal, y_test_malware))
    #df_malware.to_csv("dos_malware")
    print(df_malware.shape)

    return train_normal,x_test,y_test

if __name__ == '__main__':
    #x = np.array([[1,-1,2],
    #              [2,0,0],
    #              [0,-1,-1]])
    #x1 = get_flow_minmaxscalr(x)
    #print(x1)
    #df_normal,df_malware = process_dos()
    #combine_csv('./cicdos')
    fig_num()