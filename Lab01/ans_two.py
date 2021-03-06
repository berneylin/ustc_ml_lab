#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def load_dataset(filename):
    data_set = np.loadtxt(filename, delimiter=',')
    return data_set


def split_dataset(data_set, split_ratio=0.67):
    training_size = int(split_ratio * data_set.shape[0])
    testing_size = data_set.shape[0] - training_size
    indices_list = [x for x in range(data_set.shape[0])]
    np.random.shuffle(indices_list)
    training_set = np.ones((training_size, data_set.shape[1]))
    testing_set = np.ones((testing_size, data_set.shape[1]))

    for i in range(data_set.shape[0]):
        if i < training_size:
            training_set[i, :] = data_set[indices_list[i], :]
        else:
            testing_set[i-training_size, :] = data_set[indices_list[i], :]

    return training_set, testing_set


def separate_dataset_by_class(data_set):
    sep_dict = dict()
    for i in range(data_set.shape[0]):
        curr_data = data_set[i, :]
        if curr_data[-1] not in sep_dict:
            sep_dict[curr_data[-1]] = curr_data
        else:
            sep_dict[curr_data[-1]] = np.row_stack((sep_dict[curr_data[-1]], curr_data))

    return sep_dict


def summarize(data_set):
    summarize = []
    for i in range(data_set.shape[1] - 1):
        # 遍历data_set中的每一个特征，计算其均值和标准差
        mean = np.mean(data_set[:, i])
        std = np.std(data_set[:, i])
        if std == 0:
            print(data_set[:, i])
        summarize.append((mean, std))

    return summarize


def summarize_by_class(data_set):
    sep_dict = separate_dataset_by_class(data_set)
    summaries = dict()
    for classification, data in sep_dict.items():
        summaries[classification] = summarize(data)
    return summaries


def calc_gaussian_prob(x, mean, std):
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(std, 2))))
    return (1 / (np.sqrt(2*np.pi) * std)) * exponent


def calc_prior_by_class(training_set):
    prior_dict = dict()
    for i in range(training_set.shape[0]):
        curr_class = training_set[i, -1]
        if curr_class not in prior_dict:
            prior_dict[curr_class] = 0
        else:
            prior_dict[curr_class] += 1

    for k, v in prior_dict.items():
        prior_dict[k] = v / training_set.shape[0]

    return prior_dict


def calc_class_prob(summaries, input_vector, prior_dict):
    prob_dict = dict()
    for classification, class_summary in summaries.items():
        prob_dict[classification] = prior_dict[classification]
        for i in range(len(class_summary)):
            mean, std = class_summary[i]
            x = input_vector[i]
            prob_dict[classification] *= calc_gaussian_prob(x, mean, std)

    return prob_dict


def predict(summaries, input_vector, prior_dict):
    prob_dict = calc_class_prob(summaries, input_vector, prior_dict)
    predict_label, predict_prob = None, -1
    for classification, probability in prob_dict.items():
        if predict_label is None or probability > predict_prob:
            predict_prob = probability
            predict_label = classification

    return predict_label, predict_prob


def calc_accuracy(testing_set, summaries, prior_dict, show=True):
    acc_cnt = 0
    for i in range(testing_set.shape[0]):
        curr_data = testing_set[i, :]
        true_label = curr_data[-1]
        pred_label = predict(summaries, curr_data, prior_dict)[0]
        if true_label == pred_label:
            acc_cnt += 1
        if show:
            print('True label:', true_label, '; Predicted label:', pred_label)
    return acc_cnt/testing_set.shape[0]


def sk_learn_predict_acc(training_set, testing_set):
    from sklearn.naive_bayes import GaussianNB
    features = training_set[:, 0:-1]
    labels = training_set[:, -1]
    clf = GaussianNB()
    clf.fit(features, labels)

    true_label = testing_set[:, -1]
    pred_label = clf.predict(testing_set[:, 0:-1])
    acc_cnt = np.sum(true_label == pred_label)

    return acc_cnt/testing_set.shape[0]


if __name__ == '__main__':
    data_set = load_dataset('wine.data')

    # 因为该数据集的label放在第0列，所以需要对其进行交换
    tmp = data_set[:, -1].copy()  # numpy数据赋值引用
    data_set[:, -1] = data_set[:, 0]
    data_set[:, 0] = tmp

    training_set, testing_set = split_dataset(data_set, split_ratio=0.67)
    prior_dict = calc_prior_by_class(training_set)
    summaries = summarize_by_class(training_set)
    print("红酒数据集的预测准确率为： %.2f%%" % (calc_accuracy(testing_set, summaries, prior_dict, show=False) * 100))
    print("红酒数据集使用sci-kit-learn中的贝叶斯预测准确率为：%.2f%%" % (sk_learn_predict_acc(training_set, testing_set) * 100))
