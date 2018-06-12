#coding=utf-8
#author@zhangdong

import tensorflow as tf
from data_helper import Dataset
import numpy as np
import os,time
from para_config import config
import datetime
from char_CNN_model import charCNN

#获取训练数据
train_data=Dataset(config.train_data_source)
train_data.load_dataset()
train_x,train_y=train_data.data_x,train_data.data_y

#获取测试数据
dev_data=Dataset(config.dev_data_source)
dev_data.load_dataset()
dev_x,dev_y=dev_data.data_x,dev_data.data_y
print ('done')



with tf.Graph().as_default():
    session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess=tf.Session(config=session_config)
    with sess.as_default():
        charcnn=charCNN(config.l0,config.num_classes,config.model.conv_layers,config.model.fc_layers,l2_reg_lambda=0.0)


        global_step=tf.Variable(initial_value=0,trainable=False,name='global_step')
        #config.model.learning_rate=
        optimizer=tf.train.AdamOptimizer(config.model.learning_rate)
        grads_and_vars=optimizer.compute_gradients(charcnn.loss)
        train_op=optimizer.apply_gradients(grads_and_vars,global_step)


        #Summaries 的输出路径
        timestamp=str(int(time.time()))
        out_dir=os.path.abspath(os.path.join(os.path.curdir,'run',timestamp))
        print ('Writing to {}\n'.format(out_dir))

        #keep the track of gradient values and sparsity
        grad_summaries=[]
        for g,v in grads_and_vars:
            if g is not None:
                grad_hist_summary=tf.summary.histogram('{}/grad/hist'.format(v.name),g)
                sparsity_summary=tf.summary.scalar('{}/grad/sparsity'.format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged=tf.summary.merge(grad_summaries)

        #Summaries for  loss and accuracy
        loss_summary=tf.summary.scalar('loss',charcnn.loss)
        accuracy_summary=tf.summary.scalar('accuracy',charcnn.accuracy)

        #train summaries
        train_summary_op = tf.summary.merge([loss_summary,accuracy_summary,grad_summaries_merged])
        train_summary_dir=os.path.join(out_dir,'summaries','train')
        train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)


        #dev summaries
        dev_summary_op = tf.summary.merge([loss_summary,accuracy_summary])
        dev_summary_dir=os.path.join(out_dir,'summaries','dev')
        dev_summary_writer=tf.summary.FileWriter(dev_summary_dir,sess.graph)


        #checkpoint directory Assume this directory exists.
        checkpoint_dir=os.path.abspath(os.path.join(out_dir,'checkpoints'))
        checkpoint_prefix=os.path.join(checkpoint_dir,'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver=tf.train.Saver(tf.global_variables())

        #初始化所有变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch,y_batch):
            feed_dict={charcnn.input_x:x_batch,
                       charcnn.input_y:y_batch,
                       charcnn.dropout_keep_prob:config.model.dropout_keep_prob}

            _,step,summaries,loss,accuracy=sess.run([train_op,global_step,train_summary_op,charcnn.loss,charcnn.accuracy],feed_dict=feed_dict)
            nowtime=datetime.datetime.now().isoformat()
            print ('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime,step,loss,accuracy))
            train_summary_writer.add_summary(summaries,step)

        def dev_step(x_batch,y_batch):
            feed_dict = {charcnn.input_x: x_batch,
                         charcnn.input_y: y_batch,
                         charcnn.dropout_keep_prob: config.model.dropout_keep_prob}

            step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, charcnn.loss, charcnn.accuracy],feed_dict=feed_dict)
            nowtime = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime, step, loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)


        print ('initialization done......')

        batches = train_data.batch_iter(list(zip(train_x,train_y)), config.batch_size, num_epochs=5, shuffle=True)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % config.training.evaluate_every == 0:
                print('\nEvaluation!')
                dev_step(x_batch, y_batch)
                print('\n')

            if current_step % config.training.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))









