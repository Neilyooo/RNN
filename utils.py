#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random
import json
import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    with open('dictionary.json','r') as file :
        dictionary = json.load(file, encoding= 'utf-8')
    r"""
    参照dynamic_rnn_in_tf中的gen_batch(raw_data, batch_size, num_steps)
    dictionary('文字1':1, '文字2'：2,......)
    输入数据vocabulary中的汉字转换为数字储存
    dictionary[0]是UNK
    """
    for word in vocabulary:
        if X in dictionary.keys():
            X.append([word])
        else:
            X.append(['UNK'])
    for word in vocabulary[1:]:
        if Y in dictionary.keys():
            Y.append([word])
        else:
            Y.append(['UNK'])
    X_length = len(X)
    Y.append(0)# 添加标签Y
    r"""
    每个batchsize有多少个数据
    """
    batch_partition_length = X_length // batch_size
    r"""
    初始化数据
    """
    data_X = np.zeros([batch_size, batch_partition_length], dtype=tf.int32)
    data_Y = np.zeros([batch_size, batch_partition_length], dtype=tf.int32)
    r"""
    按batch_size分割开
    """
    for i in range(batch_size):
        data_X = X[batch_partition_length * i:batch_partition_length * (i+1)]
        data_Y = Y[batch_partition_length * i:batch_partition_length * (i+1)]
    epoch_size = batch_partition_length // num_steps
    r"""
    返回X,Y
    """
    for i in range(epoch_size):
        x = data_X[:, i*num_steps:(i+1)*num_steps]
        y = data_Y[:, i*num_steps:(i+1)*num_steps]
        yield (x,y)
    ##################
    # Your Code here
    ##################


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
