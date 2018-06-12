#coding=utf-8
#author@zhangdong

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


def length(sequences):
#求序列的每个元素的长度，比如对于[64,30,30]的序列，要求64个元素的长度
    used=tf.sign(tf.reduce_max(tf.abs(sequences),reduction_indices=2))
    seq_len=tf.reduce_sum(used,reduction_indices=1)
    return tf.cast(seq_len,tf.int32)




class HAN(object):

    #为了方便理解tensor在各个过程中的变化情况
    #令batch_size=64,max_sentence_num=30,max_sentence_length=30,embedding_size=200

    def __init__(self,vocab_size,num_classes,embedding_size=200,hidden_size=50):
        self.vocab_size=vocab_size
        self.num_classes=num_classes
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size

        with tf.name_scope('placeholder'):
            self.max_sentence_num=tf.placeholder(tf.int32,name='max_sentence_num')
            self.max_sentence_length=tf.placeholder(tf.int32,name='max_sentence_length')
            self.batch_size=tf.placeholder(tf.int32,name='batch_size')
            #input的shape是[batch_size,句子数，句子中单词数]，由于每个样本不一样，所以指定None
            self.input_x=tf.placeholder(tf.int32,shape=[None,None,None],name='input_x')
            self.input_y=tf.placeholder(tf.float32,shape=[None,num_classes],name='input_y')


        #构建模型，整个模型的流程
        word_embedded=self.word2vec()
        sent_vec=self.sent2vec(word_embedded)
        doc_vec=self.doc2vec(sent_vec)
        out=self.classifer(doc_vec)

        self.out=out








    def word2vec(self):
        #嵌入层，这里初始化词向量
        with tf.name_scope('word_embedding'):
            embedding_mat=tf.Variable(tf.truncated_normal((self.vocab_size,self.embedding_size)))     #初始化单词向量矩阵，[vocab_size,embedding_size]
            word_embedded=tf.nn.embedding_lookup(embedding_mat,self.input_x)            #shape为[64,30,30,200]
            return word_embedded

    def sent2vec(self,word_embedded):
        with tf.name_scope('sent2vec'):
            #注意：在双向RNN或者GRU中，有一个参数major_time,它的bool值决定了输入tensor的shape。
            #major_time默认是False，当False时，输入tensor必须是[bacth_size,max_time,depth]
            #当为True的时候，输入tensor必须是[max_time,batch_size,depth]
            #注意到，输入tensor必须是三维的，所以首先要reshape输入
            word_embedded=tf.reshape(word_embedded,[-1,self.max_sentence_length,self.embedding_size])
            #shape为[batch_size*max_sent_num,max_sent_length,embedding_size] [64*30,30,200]

            word_encoded=self.BidirectionalGRUEncoder(word_embedded,name='word_encoded')
            #shape为[batch_size*max_sent_num,max_sent_length,hidden_size*2]   [64*30,30,50*2]

            sent_vec=self.AttentionLayer(word_encoded,name='word_attention')
            #经过attention后为[batch_size*max_sent_num,hidden_size*2]       [64*30,100]

            return sent_vec





    def doc2vec(self,sent_vec):
        with tf.name_scope('doc2vec'):
            #由于得到的sent_vec是[64*30,100]是二维的，该函数还是要通过双向GRU,所以还是转化为三维的，首先要reshape
            sent_vec=tf.reshape(sent_vec,shape=[-1,self.max_sentence_num,self.hidden_size*2])
            #shape为[64,30,100]

            doc_encoded=self.BidirectionalGRUEncoder(sent_vec,name='sent_encoded')
            #shape为[64,30,100]

            doc_vec=self.AttentionLayer(doc_encoded,name='sent_attention')
            #shape为[batch_size,hidden_size*2]       [64,100]

            return doc_vec




    def classifer(self,doc_vec):
        #得到doc_vec后，通过一个全连接层，得到类别
        with tf.name_scope('doc_classification'):
            out=layers.fully_connected(inputs=doc_vec,num_outputs=self.num_classes,activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self,inputs,name):
        #把输入的单词向量或者句子向量编码成hidden*2的向量
        with tf.variable_scope(name):
            GRU_cell_fw=rnn.GRUCell(self.hidden_size)       #前向GRU
            GRU_cell_bw=rnn.GRUCell(self.hidden_size)       #后向GRU

            #通过查看birnn源码，可以看到它的返回值
            (output_fw, output_bw),(output_state_fw, output_state_bw)=tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,cell_bw=GRU_cell_bw,inputs=inputs,sequence_length=length(inputs),dtype=tf.float32)
            #output_bw,output_fw的shape都是[batch_size,max_time,hidden_size]

            output_concat=tf.concat((output_fw,output_bw),2)    #shape是[batch_size,max_time,hidden_size*2]

            return output_concat


    def AttentionLayer(self,inputs,name):
        #AttentionLayer层的输入是双向GRU的输出，则inputs的维度是[batch_size,max_time,hidden_size*2]
        with tf.variable_scope(name):
            #需要定义一个代表上下文的权重向量，u_context是上下文的重要性向量，用于区分不同单词对句子的重要程度，不同句子对文章的重要程度
            #因为需要对biGRU的输出进行编码，所以其u_context的长度为[hidden_size*2]
            u_context=tf.Variable(tf.truncated_normal(shape=[self.hidden_size*2]),name='u_context')

            h=layers.fully_connected(inputs=inputs,num_outputs=self.hidden_size*2,activation_fn=tf.nn.tanh)
            #h是[batch_size,max_time,hidden_size*2]

            alpha=tf.nn.softmax(tf.reduce_sum(tf.multiply(h,u_context),axis=2,keep_dims=True),dim=1)
            #softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)          #正好对应论文中alpha的求解公式
            #alpha的维度是[batch_size,max_time,1]


            attention_output=tf.reduce_sum(tf.multiply(inputs,alpha),axis=1)

            return attention_output



