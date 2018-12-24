#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    @description: Perceptron algorithm realized in Adult dataset.
    @author: Bernard Lin
    @date  : 2018-12-21
    @email : nardlin@mail.ustc.edu.cn
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


def deal_data(filename):
    # 从数据集中获得原始数据
    adult_raw = pd.read_csv(filename, header=None)
    # print(len(adult_raw))
    # 添加标题
    adult_raw.rename(columns={0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education', 4: 'education_number',
                              5: 'marriage', 6: 'occupation', 7: 'relationship', 8: 'race', 9: 'sex',
                              10: 'capital_gain', 11: 'apital_loss', 12: 'hours_per_week', 13: 'native_country',
                              14: 'income'}, inplace=True)
    # 清理数据，删除缺失值
    adult_cleaned = adult_raw.dropna()

    # 属性数字化
    adult_digitization = pd.DataFrame()
    target_columns = ['workclass', 'education', 'marriage', 'occupation', 'relationship', 'race', 'sex',
                      'native_country',
                      'income']
    for column in adult_cleaned.columns:
        if column in target_columns:
            unique_value = list(enumerate(np.unique(adult_cleaned[column])))
            dict_data = {key: value for value, key in unique_value}
            adult_digitization[column] = adult_cleaned[column].map(dict_data)
        else:
            adult_digitization[column] = adult_cleaned[column]
    # 确认数据类型为int型数据
    # for column in adult_digitization:
    #     adult_digitization[column] = adult_digitization[column].astype(int)
    # adult_digitization.to_csv("data_cleaned.csv")
    # print(len(adult_cleaned))
    # 构造输入和输出
    X = adult_digitization[
        ['age', 'workclass', 'fnlwgt', 'education', 'education_number', 'marriage', 'occupation', 'relationship',
         'race',
         'sex', 'capital_gain', 'apital_loss', 'hours_per_week', 'native_country']]
    Y = adult_digitization[['income']]
    # 查看数据情况 0:22654, 1:7508
    # print(Y.value_counts())
    # （0.7:0.3）构造训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)

    return np.array(X_train, dtype=np.float64), np.array(X_test, dtype=np.float64), \
           np.array(Y_train, dtype=np.float64), np.array(Y_test, dtype=np.float64)


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = deal_data('./adult.data')
    sc = StandardScaler()
    sc.fit(X_train)
    Y_train = np.ravel(Y_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(tol=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, Y_train)
    y_pred = ppn.predict(X_test_std)

    print(np.sum(Y_test.ravel() == y_pred) / Y_test.shape[0])
