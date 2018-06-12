#coding=utf-8
#author@zhangdong



import tensorflow as tf
import numpy as np

def length(sequences):
#返回一个序列中每个元素的长度
#输入的序列是三维的
#以[64,30,30]的序列为例
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
            #首先求绝对值，再按照第三个维度求最大值，得到[64,30]的矩阵，再进行tf.sign操作，用1，0，-1分别代表大于等于小于0的数
            #因为已经求过绝对值，所以基本上是tf.sign之后基本上是1
    print (used.shape,used.eval())
    seq_len = tf.reduce_sum(used, reduction_indices=1)
            #再按照第二个维度进行维度上的求和
    print (seq_len.shape,seq_len.eval())
    return tf.cast(seq_len, tf.int32)       #再把元素换成int类型

a=np.ones([64,30,30])
#a=np.array([[1,2,-3,-4,5],[1,2,3,4,5]])


m=np.array([[1,2,3],[1,2,3],[1,2,3]])
n=np.array([[2],[2],[2]])
with tf.Session() as sess:

    #re=length(a)

    #print (re.shape,re.eval())
    r=tf.multiply(m,n)
    print (r.eval())