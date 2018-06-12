#__author__=="zhangdong"
#@zhangdong

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    '''
    我们假设相关参数的值，进行卷积过程中数据shape的变化演示
    batch_szie=64
    sequence_szie=50
    embedding_szie=128
    num_filters=128
	 filter_size='3,4,5'   
	 vocab_size=10000 
    '''

    def __init__(
            self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
            l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # 输入
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # 输入的标签
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # dropout?

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)  # 先初始化为0，后面可以加上最后一层的W,b的l2正则惩罚项损失

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                 name="W")  # 初始化权重矩阵[vocab_size, embedding_size].[10000,128]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # 取相对应的向量   [64,50,128]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # [64,50,128,1]

        # 创建一个卷积层和最大池化层
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1,
                                num_filters]  # 卷积核的shape是[3,128,1,128]/[4,128,1,128]/[5,128,1,128]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")  # 卷积核的权重  [3,128,1,128]/[4,128,1,128]/[5,128,1,128]
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")  # 偏置		[128]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               name="relu")  # 卷积后的操作结果是[64,48,1,128]/[64,47,1,128]/[64,46,1,128]

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")  # max-pooling后的结果[64,1,1,128]/[64,1,1,128]/[64,1,1,128]
                pooled_outputs.append(pooled)

        # max-pooling结束后得到pooled_outputs=[[64,1,1,128],[64,1,1,128],[64,1,1,128]]

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)  # 394
        self.h_pool = tf.concat(pooled_outputs, 3)  # 在第四个维度上连接[64,1,1,394]，这样每个batch中的64个文本，被转换为一个384维向量
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # 再把[64,1,1,394]reshape成[64,384]的矩阵

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)  # dropout操作后还是[64,384]

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())#tf.contrib.layers.xavier_initializer()这是一种经典的权重初始化方法
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")  # x*w+b的结果，这里没有经过sigmoid或者其他激活函数   [64,5]
            self.predictions = tf.argmax(self.scores, 1, name="predictions")  # 把第二维度中的最大值索引返回，也就是5维向量中最大元素的索引

        # Calculate mean cross-entropy loss			#计算交叉熵
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # 对正确标签向量，取出最大元素的索引，并看与预测的是否相等tf.equal,返回bool数组，如某个batch中64个有60个预测正确[True,True,True,True,True,..有60个True,4个False]
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # 把64维bool数组correct_predictions  tf.cast为float类型；在求平均值，就是60.0/64,得到的就是accuracy