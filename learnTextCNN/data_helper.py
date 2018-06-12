#__author__=="zhangdong"
#@zhangdong


import re
import numpy as np


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


def load_data_and_labels(raw_file):

    label_dict={1:[1,0,0,0,0],2:[0,1,0,0,0],3:[0,0,1,0,0],\
                4:[0,0,0,1,0],5:[0,0,0,0,1]}
    with open(raw_file,'r') as fr:
        x_text=[];y=[]
        for line in fr.readlines():
            line=line.strip().split('\t')
            text=clean_text(line[0])        #获取文本数据
            label=line[1]                   #获取Label
            x_text.append(text)
            y.append(label_dict[int(label)])
        y=np.array(y,dtype=float)

    return [x_text,y]



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






#all_data=load_data_and_labels('../yelp_text_star.txt')

#print(all_data[0][1])
