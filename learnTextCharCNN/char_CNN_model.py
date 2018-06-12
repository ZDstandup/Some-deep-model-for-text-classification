#coding=utf-8
#author@zhangdong

import tensorflow as tf
import numpy as np
from math import sqrt
from data_helper import Dataset
from para_config import config

class charCNN(object):

    #为了便于理解，以batch_size=128为例，来说明tensor 的shape的变化过程

    '''
    该类是char-CNN模型，用于文本分类
    '''
    def __init__(self,l0,num_classes,conv_layers,fc_layers,l2_reg_lambda=0.0001):

        #placeholders for input_x,input_y and droupout
        self.input_x=tf.placeholder(dtype=tf.int64,shape=[None,l0],name='input_x')
        self.input_y=tf.placeholder(dtype=tf.float32,shape=[None,num_classes],name='input_y')
        self.dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='dropout_keep_prob')

        #keeping track of l2 regularization loss (optional)
        l2_losses=tf.constant(0.0)

        #Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            train_data=Dataset(config.train_data_source)
            self.char_embedding_mat,self.char_embedding_dict=train_data.onehot_dict_build()
            self.input_x_vec=tf.nn.embedding_lookup(self.char_embedding_mat,self.input_x)           #获取当前batch的input_x的索引向量     [128,1014,69]
            self.input_x_expanded=tf.expand_dims(self.input_x_vec,-1)               #多加一个维度，变成4维的           [128,1014,69,1]
            self.x_flat=self.input_x_expanded



        #convolutional layers
        #卷积层的输入是self.input_x_expanded   [128,1014,69,1]
        for i , conv in enumerate(conv_layers):
            with tf.name_scope('no.%d_conv_layer'%(i+1)):
                print ('start to process conv layer-%s'%(str(i+1)))
                filter_width=self.x_flat.get_shape()[2].value
                #由于每次卷积的结果都在变化，所以每次卷积处理的维度或者区域是不同的，也就是filter的width不一样，但是width都是get_shape()[2].value

                # conv_layers = [[256, 7, 3],
                #                [256, 7, 3],
                #                [256, 3, None],
                #                [256, 3, None],
                #                [256, 3, None],
                #                [256, 3, 3]]
                filter_shape = [conv[1],filter_width,1,conv[0]]
                #filter_weights=tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=0.05),name='filter_weights')
                stdv = 1 / sqrt(conv[0] * conv[1])
                filter_weights =tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv),dtype='float32', name='w')
                #下面是另一种权重初始化方式

                #b=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[conv[0]]),name='b')
                b = tf.Variable(tf.random_uniform(shape=[conv[0]], minval=-stdv, maxval=stdv), name='b')


                #下面是卷积操作
                conv_results=tf.nn.conv2d(self.x_flat,filter=filter_weights,strides=[1,1,1,1],padding="VALID",name='conv')

                h_conv=tf.nn.bias_add(conv_results,bias=b)

                if  conv[-1] is not None:    #如果需要迟化操作
                    ksize=[1,conv[-1],1,1]
                    pool_conv=tf.nn.max_pool(h_conv,ksize=ksize,strides=ksize,padding='VALID',name='max-pooling')
                else:
                    pool_conv=h_conv        #不进行池化操作


                self.x_flat=tf.transpose(pool_conv,perm=[0,1,3,2])

        #当前循环，6个卷积层结束,最后获得[128,34,256,1]的矩阵，在通过全连接层时需要转换矩阵的维度，转换成128个34*256的特征向量


        with tf.name_scope('reshape'):
            first_fc_input_dim=self.x_flat.get_shape()[1].value*self.x_flat.get_shape()[2].value     #第一层全连接层的输入维度
            self.x_flat=tf.reshape(self.x_flat,[-1,first_fc_input_dim])

        #full_connected layer
        weights_dims=[first_fc_input_dim]+fc_layers
        for j , fc in enumerate(fc_layers):
            with tf.name_scope('no.%s_fc_layer'%(str(j+1))):
                print('start to process the no.%d fc layer'%(j+1))
                stdv = 1 / sqrt(weights_dims[j])
                fc_weights = tf.Variable(tf.random_uniform(shape=[weights_dims[j],fc], minval=-stdv, maxval=stdv), dtype='float32', name='w')
                # fc_weights=tf.Variable(tf.truncated_normal(shape=[weights_dims[j],fc],stddev=0.05),name='fc_weights')
                # fc_b=tf.Variable(tf.constant(0.1,shape=[fc]),name='fc_b')

                fc_b = tf.Variable(tf.random_uniform(shape=[fc], minval=-stdv, maxval=stdv), dtype='float32', name='b')

                self.x_flat=tf.nn.relu(tf.matmul(self.x_flat,fc_weights)+fc_b)

                with tf.name_scope('dropout'):
                    tf.nn.dropout(self.x_flat,keep_prob=self.dropout_keep_prob)

        #此时当前循环下的前两个全连接层结束，后面是输出

        with tf.name_scope('output'):
            # outlayer_weights=tf.Variable(tf.truncated_normal(shape=[fc_layers[-1],num_classes],stddev=0.05),name='outlayer_weights')
            # outlayer_b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name='outlayer_b')

            stdv = 1 / sqrt(weights_dims[-1])
            outlayer_weights = tf.Variable(tf.random_uniform([fc_layers[-1], num_classes], minval=-stdv, maxval=stdv),dtype='float32', name='outlayer_weights')
            outlayer_b = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stdv, maxval=stdv), name='outlayer_b')

            l2_losses += tf.nn.l2_loss(outlayer_weights)
            l2_losses+=tf.nn.l2_loss(outlayer_b)
            self.y_pred=tf.nn.xw_plus_b(self.x_flat,outlayer_weights,outlayer_b,name='y_pred')
            self.predictions=tf.argmax(self.y_pred,1,name='predictions')


        #计算平均交叉熵
        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.y_pred)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_losses         #平均交叉熵

        with tf.name_scope('accuracy'):
            self.correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(self.correct_predictions,'float'),name='accuracy')



