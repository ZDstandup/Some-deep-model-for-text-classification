#__author__=="zhangdong"
#@zhangdong

import tensorflow as tf
import numpy as np

class TextCNN(object):

    #构造函数，里面的操作都是自动执行
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):



        #1.placeholder for input , and dropout
        self.input_x = tf.placeholder(tf.int32,shape=[None,sequence_length],name='input_x')         #占位符   得到的元素是词的序号
        self.input_y = tf.placeholder(tf.float32,shape=[None,num_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')



        #(optional : keep tghe track of l2 regularization loss)
        #可选项，可加可不加，作用不大
        l2_loss=tf.constant(0.0)


        #2.embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W=tf.Variable(tf.random_normal([vocab_size,embedding_size],-1.0,1.0),name='W')
            self.embedding_chars=tf.nn.embedding_lookup(self.W,self.input_x)    #从W中取出与input_x中对应的序号的初始化向量
            #embedding_chars是batch_size*sequenceze_length*embedding_size的矩阵
            self.embedding_chars_expanded=tf.expand_dims(self.embedding_chars,-1)
            #embedding_chars_expanded是batch_size*sequenceze_length*embedding_size*1的矩阵

            '''
                #这样就得到每一个batch中 每个文本中每个词的向量
                #最后的输入是batch_size*sequenceze_length*embedding_size*1的矩阵
           '''

        #3.create a convolution+max-pooling layer for each filter size
        pooled_outputs=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'%(filter_size)):
                #对于每种size的卷积
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W=tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=1.0),name="W")
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')
                conv=tf.nn.conv2d(input=self.embedding_chars_expanded,
                    filter=W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv')

                #对卷积的结果通过非线性激活函数
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')

                pooled=tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')

                pooled_outputs.append(pooled)

        #4.Combine all  the pooled features
        # max-pooling结束后得到pooled_outputs=[[64,1,1,128],[64,1,1,128],[64,1,1,128]]
        # 把以上的特征连接在一起
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        #5.Add dropout
        with tf.name_scope('dropout'):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)



        #6.Final (unormalized)  scores and predictions
        with tf.name_scope('output'):
            W=tf.get_variable('W',shape=[num_filters_total,num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name='b')
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)
            self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name='scores')
            self.predictions=tf.argmax(self.scores,1,name='predictions')


        #7.Calculate  mean cross-entropy loss
        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

        #8. Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name='accuracy')

