#__author__=="zhangdong"
#@zhangdong

import tensorflow as tf
import numpy as np

batch_szie=64
sequence_length=50
vocab_size=10000
embedding_size=128

input_x=[[1,2],[3,4],[5,6]]
W=tf.Variable(tf.random_normal([vocab_size,embedding_size],-1.0,1.0),name='W')
embedding_chars=tf.nn.embedding_lookup(W,input_x)    #从W中取出与input_x中对应的序号的初始化向量
embedding_chars_expanded=tf.expand_dims(embedding_chars,-1)

x=tf.ones((3,2,2))
C=[x,x,x]
print(tf.concat(C,2).shape)
#M=tf.Variable(tf.random_normal([64,1,1,394],-1.0,1.0),name='W')
M=np.array([[[[1,2,3,22]]],[[[4,5,6,2]]],[[[7,8,9,3]]]],dtype=float)


with tf.Session() as sess:
    all_inits=tf.global_variables_initializer()
    print(W.shape)
    print(embedding_chars.shape)
    print (embedding_chars_expanded.shape)
    print(tf.concat(C,2).shape)
    print(M.shape)
    M1=tf.reshape(M,[2,6])
    print (M1.shape)
    print (M1.eval())
    M_dropout=tf.nn.dropout(M1, 0.5)
    print (M_dropout.shape)
    print (M_dropout.eval())


