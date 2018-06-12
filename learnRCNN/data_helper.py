#coding=utf-8
#author@zhangdong

import re
import numpy as np
import csv

# source='./../data/ag_news_csv/train.csv'
#
# csvfile=open(source,'r')
# for line in csv.reader(csvfile,delimiter=',',quotechar='"'):
#     content=line[1]+'. '+line[2]
#     label=line[0]


def load_dataset(data_source):
    #把原始的csv文件处理成字符的索引形式的矩阵，也就是处理成data_size*1014的矩阵
    '''
            处理过程中需要的函数有:1.创建字符字典和字符对应的one-hot向量矩阵，这个矩阵是留在train时候embedding layer中look_up的。
                                    2.需要把每个文本转化为一个1014的向量。向量元素是字符索引
    '''
    docs=[]     #用来存储所有的原始文本
    doc_count=0         #用来记录数据集中文本总数
    labels=[]           #用来存放每个文本的标签类别

    label_oht=np.eye(4)

    csvfile=open(data_source,'r')
    for line in csv.reader(csvfile,delimiter=',',quotechar='"'):
        content=line[1]+'. '+line[2]
        content=clean_text(content)
        label=int(line[0])
        one_hot=label_oht[label-1]
        doc_count+=1
        docs.append(content)
        labels.append(one_hot)

    return docs,labels

def clean_text(string):
    """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def batch_iter( data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min(batch_size * (batch_num + 1), data_size)
            yield shuffled_data[start_index:end_index]


# docs,labels=load_dataset('./../data/ag_news_csv/train.csv')
#
# print (labels)