#coding=utf-8
#author@zhangdong

import nltk
from nltk.tokenize import WordPunctTokenizer
import os
from collections import defaultdict
import pickle
import json
import numpy as np



sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')      #分句
word_tokenizer=WordPunctTokenizer()         #分词

def build_vocab(vocab_path,yelp_filename):

    if os.path.exists(vocab_path):
        vocab_file=open(vocab_path,'rb')
        vocab=pickle.load(vocab_file)
        print ('load vocab finished!')
    else:

        word_freq=defaultdict(int)
        with open(yelp_filename,'r') as fr:
            for line in fr:
                review=json.loads(line)
                words=word_tokenizer.tokenize(review['text'])
                for word in words:
                    word_freq[word]+=1
            print ('load finished')


        #把词频小于5的word看为UNK,在词典中序号都为0
        vocab={}
        vocab['UNKNOW_TOKEN']=0
        i=1
        for word,fre in word_freq.items():
            if fre>5:
                vocab[word]=i
                i+=1

        #保存这个词典
        with open(vocab_path,'wb') as f:
            pickle.dump(vocab,f)
            print (len(vocab))      #8783
            print ('vocab saved !')

    return vocab


#vocab=build_vocab('yelp_academic_dataset_review_vocab.pk','./../yelp_academic_dataset_review.json')



def load_dataset(yelp_filename,max_sent_in_doc=30,max_word_in_sent=30,dev_percent=0.1):
    yelp_data_path='yelp_data.pk'
    vocab_path='yelp_academic_dataset_review_vocab.pk'
    doc_num=8094        #只使用8094个样本

    if not os.path.exists(yelp_data_path):
        vocab=build_vocab(vocab_path,yelp_filename)
        num_classes=5
        UNKNOW=0
        data_x=np.ones([doc_num,max_sent_in_doc,max_word_in_sent])
        data_y=[]

        with open(yelp_filename,'r') as fr:
            for line_index,line in enumerate(fr):
                review=json.loads(line)
                sents=sent_tokenizer.tokenize(review['text'])       #把样本text分句子

                doc = np.zeros([max_sent_in_doc, max_word_in_sent])

                for i , sent in enumerate(sents):
                    if i < max_sent_in_doc:
                        word_to_index=np.zeros([max_word_in_sent],dtype=int)      #每个句子的表示
                        for j,word in enumerate(word_tokenizer.tokenize(sent)):
                            if j < max_word_in_sent:
                                word_to_index[j]=vocab.get(word,UNKNOW)         #获取这个word的序号，没有的话就是默认的UNKNOW=0

                        doc[i]=word_to_index        #第i个句子就有word_to_index表示


                data_x[line_index]=doc      #data_x的数据集中的第line_index行就是这个doc表示了

                #---下面获取label
                label=int(review['stars'])
                label_vec=[0]*num_classes
                label_vec[label-1]=1        #用一个5维的向量表示标签

                data_y.append(label_vec)        #添加到data_y中

                print(line_index)

            #此时当前for循环已经构建好data_x与data_y
            #然后保存起来，保存到yelp_data_path
            pickle.dump((data_x,data_y),open(yelp_data_path,'wb'))
            print ('总样本数：',len(data_x))

    else:
        pkfile=open(yelp_data_path,'rb')
        data_x,data_y=pickle.load(pkfile)


    #划分数据集
    #进行shuffle
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(data_y)))
    print (shuffle_indices)
    x_shuffled = data_x[shuffle_indices]
    y_shuffled = np.array(data_y)[shuffle_indices]

    # split train/test set

    dev_index=-1*int(dev_percent*len(data_y))
    train_x, train_y = x_shuffled[:dev_index], y_shuffled[:dev_index]
    dev_x, dev_y = x_shuffled[dev_index:], y_shuffled[dev_index:]

    return train_x,train_y,dev_x,dev_y


#train_x,train_y,dev_x,dev_y=load_dataset('./../yelp_academic_dataset_review.json',30,30)



def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)
    data_size=len(data)
    num_batches_per_epoch=int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data

        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min(batch_size*(batch_num+1),data_size)
            yield shuffled_data[start_index:end_index]






