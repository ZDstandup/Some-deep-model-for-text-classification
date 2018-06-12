#coding=utf-8
#author@zhangdong

from para_config import config          #导入模型，训练过程的参数
import numpy as np
import csv

# source='./../data/ag_news_csv/train.csv'
#
# csvfile=open(source,'r')
# for line in csv.reader(csvfile,delimiter=',',quotechar='"'):
#     content=line[1]+'. '+line[2]
#     label=line[0]


class Dataset(object):
    def __init__(self,data_source):
        self.data_source=data_source
        self.alphalet=config.alphalet
        self.alphalet_size=config.alphalet_size
        self.num_classes=config.num_classes
        self.l0=config.l0
        self.batch_size=config.batch_size
        self.data_x=[]
        self.data_y=[]

    def batch_iter(self,data, batch_size, num_epochs, shuffle=True):
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

    def load_dataset(self):
        #把原始的csv文件处理成字符的索引形式的矩阵，也就是处理成data_size*1014的矩阵
        '''
            #处理过程中需要的函数有:1.创建字符字典和字符对应的one-hot向量矩阵，这个矩阵是留在train时候embedding layer中look_up的。
                                    2.需要把每个文本转化为一个1014的向量。向量元素是字符索引
        '''
        docs=[]     #用来存储所有的原始文本
        doc_count=0         #用来记录数据集中文本总数
        labels=[]           #用来存放每个文本的标签类别

        csvfile=open(self.data_source,'r')
        for line in csv.reader(csvfile,delimiter=',',quotechar='"'):
            content=line[1]+'. '+line[2]
            label=int(line[0])
            doc_count+=1
            docs.append(content.lower())
            labels.append(label)

        print ('引入字符索引字典和字符oht编码......')
        char_embedding_mat,char_embedding_dict=self.onehot_dict_build()


        data_x=[]       #用来存放文本索引向量
        data_y=[]       #用来存放标签向量
        print ('start to process each doc....')

        #这里有一个把文本doc处理成字符索引组成的向量的函数doc_process()

        for i in range(doc_count):
            doc_vec_i=self.doc_process(docs[i],char_embedding_dict)             #求文本的字符索引向量
            data_x.append(doc_vec_i)

            label_vec_i=np.zeros(shape=self.num_classes,dtype='float32')         #求文本的标签向量
            label_vec_i[labels[i]-1]=1
            data_y.append(label_vec_i)


        del char_embedding_dict,char_embedding_mat
        print('求的训练集或者验证集......')
        self.data_x=np.asarray(data_x,dtype='int64')
        self.data_y=np.array(data_y,dtype='float32')






    def doc_process(self,doc,char_embedding_dict):
        min_len=min(len(doc),self.l0)
        doc_vec=np.zeros(self.l0,dtype='int64')

        #那些多出的长度都为0
        for j in range(min_len):
            if doc[j] in char_embedding_dict:
                doc_vec[j]=char_embedding_dict[doc[j]]
            else:
                doc_vec[j]=char_embedding_dict['UNK']
        return doc_vec      #返回文本的索引向量

    def onehot_dict_build(self):
        char_embedding_dict={}
        char_embedding_dict['UNK']=0
        char_embedding_mat=[]
        char_embedding_mat.append(np.zeros(self.alphalet_size,dtype='float32'))     #首先把69维全0向量加进去，也就是UNK的one-hot-coding
        for i ,char in enumerate(self.alphalet):
            onehot=np.zeros(self.alphalet_size,dtype='float32')     #初始化一个69维向量
            char_embedding_dict[char]=i+1
            onehot[i]=1
            char_embedding_mat.append(onehot)               #把当前字符char的编码加入矩阵中

        char_embedding_mat=np.array(char_embedding_mat)
        return char_embedding_mat,char_embedding_dict       #返回字符向量矩阵和字符字典




# trainData=Dataset(config.train_data_source)
#
# trainData.load_dataset()
#
# batches=trainData.batch_iter(list(zip(trainData.data_x,trainData.data_y)),trainData.batch_size,num_epochs=10,shuffle=True)
#
# for batch in batches:
#     input_x,input_y=zip(*batch)
#     input_x=[list(x) for x in input_x]
#     print (input_x)
#     break
#
#
# import tensorflow as tf
# with tf.Session() as sess:
#     train_data = Dataset(config.train_data_source)
#     W, _ = train_data.onehot_dict_build()
#     x_image = tf.nn.embedding_lookup(W, input_x)
#     x_flat = tf.expand_dims(x_image, -1)
#     filter_width = x_flat.get_shape()[2].value
#     print (filter_width)