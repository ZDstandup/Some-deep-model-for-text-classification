#__author__=="zhangdong"
#@zhangdong

import tensorflow as tf
import numpy as np
import data_helper
from text_cnn import TextCNN
from tensorflow.contrib import learn
import os,time,datetime


#---参数---
#=========================================


#parameters when loading data
tf.flags.DEFINE_string('raw_file','../yelp_text_star.txt','data source for all classes data')
tf.flags.DEFINE_float('dev_sample_percentage',0.1,'percentage of allData as dev')

#parameters of the model
tf.flags.DEFINE_integer('embedding_dim',128,'word embedding size (default 128)')
tf.flags.DEFINE_string('filter_sizes','3,4,5','filter sizes default (3,4,5)')
tf.flags.DEFINE_integer('num_filters',128,'Numbers of filters per filter size (default:128)')
tf.flags.DEFINE_float('dropout_keep_prob',0.5,'dropout keep probability (default:0.5)')
tf.flags.DEFINE_float('l2_reg_lambda',0.0,'L2 regularization lambda (default:0.0)')


#Training parameters
tf.flags.DEFINE_integer('batch_size',64,'the batch size default:64')
tf.flags.DEFINE_integer('num_epochs',10,'Number of training epochs default: 200')
tf.flags.DEFINE_integer('evaluation_every',100,'avaluate on dev set every this many steps default 100')
tf.flags.DEFINE_integer('checkpoint_every',100,'save the model after this many steps default:100')
tf.flags.DEFINE_integer('num_checkpoints',5,'Numbers of chechpoints to store defaul:5')         #只保存5次


#Mics parameters/ Other parameters
tf.flags.DEFINE_boolean('log_device_placement',False,'log placement of ops on devices')#是否打印设备分配的日志
tf.flags.DEFINE_boolean('allow_soft_placement',True,'allow TF soft placement')          #如果指定的设备不存在，允许TF自动分配设备


FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print ('all related parameters in CNN:')
for attr,value in sorted(FLAGS.__flags.items()):
    print ('{}={}'.format(attr.upper(),value))


print ('参数打印完毕.....')



#load data
x_text,y=data_helper.load_data_and_labels(FLAGS.raw_file)

#build vocabulary
max_document_length=max([len(x.split(' ')) for x in x_text])        #1078有点长
vocab_processor=learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length,min_frequency=3)
x=np.array(list(vocab_processor.fit_transform(x_text)))


#randomly shuffle the data
np.random.seed(10)
shuffle_indices=np.random.permutation(np.arange(len(y)))
x_shuffled=x[shuffle_indices]
y_shuffled=y[shuffle_indices]




#split train/test set
dev_sample_index=-1*int(FLAGS.dev_sample_percentage*len(y))
#print (dev_sample_index)
x_train,y_train=x_shuffled[:dev_sample_index],y_shuffled[:dev_sample_index]
x_dev,y_dev=x_shuffled[dev_sample_index:],y_shuffled[dev_sample_index:]

print ('Vocabulary size : {}'.format(len(vocab_processor.vocabulary_)))
print ('train/dev : {}/{}'.format(len(x_train),len(x_dev)))


#training process

with tf.Graph().as_default():
    session_config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                                  allow_soft_placement=FLAGS.allow_soft_placement)
    sess=tf.Session(config=session_config)
    with sess.as_default():
        cnn=TextCNN(sequence_length=len(x_train[0]),
                    num_classes=len(y_train[0]),
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int,FLAGS.filter_sizes.split(','))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)


        # Define training procedures
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 迭代步数，会自动加一
        optimizer = tf.train.AdamOptimizer(0.001)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op=optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=global_step)

        timestamp=str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        #存储模型
        checkpoint_dir=os.path.abspath(os.path.join(out_dir,'checkpoints'))
        checkpoint_prefix=os.path.join(checkpoint_dir,'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver=tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)


        #write vocabulary

        vocab_processor.save(os.path.join(out_dir,'vocab'))



        #初始化所有变量
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch,y_batch):
            #single  train  step
            feed_dict={cnn.input_x:x_batch,
                       cnn.input_y:y_batch,
                       cnn.dropout_keep_prob:FLAGS.dropout_keep_prob}
            _,step,loss,accuracy=sess.run([train_op,global_step,cnn.loss,cnn.accuracy],feed_dict=feed_dict)
            time_str=datetime.datetime.now().isoformat()

            print ("{}: step {:d}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy))

        def dev_step(x_batch,y_batch):
            feed_dict = {cnn.input_x: x_batch,
                         cnn.input_y: y_batch,
                         cnn.dropout_keep_prob: 1.0}
            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {:d}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        #generate batches

        batches=data_helper.batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size,FLAGS.num_epochs,shuffle=True)


        #整个训练过程
        for batch in batches:#[(x1,y1),(x2,y2),...,(x64,y64)]
            x_batch,y_batch=zip(*batch)     #[x1,x2,...x64]  [y1,y2,...,y64]
            train_step(x_batch,y_batch)
            current_step=tf.train.global_step(sess,global_step)

            if current_step%FLAGS.evaluation_every==0:
                print("\nEvaluation....")
                dev_step(x_dev,y_dev)
                print ('\n')

            if current_step%FLAGS.checkpoint_every==0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))






