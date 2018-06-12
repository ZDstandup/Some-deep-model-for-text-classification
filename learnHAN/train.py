#coding=utf-8
#author@zhangdong

import tensorflow as tf
import data_helper
import HAN_model
import time
import os


#参数介绍
#加载数据的参数
tf.flags.DEFINE_string('raw_file','./../yelp_academic_dataset_review.json','data source for training!')
tf.flags.DEFINE_integer('vocab_size',8783,'the size of the data word vocab')

#模型的参数
tf.flags.DEFINE_integer('num_classes',5,'the num of the classes')
tf.flags.DEFINE_integer('embedding_size',200,'size of the word embedding')
tf.flags.DEFINE_integer('hidden_size',50,'the hidden size of the biGRU')


#训练的参数
tf.flags.DEFINE_integer('batch_size',64,'batch size')
tf.flags.DEFINE_integer('num_epochs',10,'the eopchs of the training')
tf.flags.DEFINE_integer('checkpoint_every',100,'save the model after this steps')
tf.flags.DEFINE_integer('num_checkpoints',5,'Number of the checkpoint to store default:5')
tf.flags.DEFINE_integer('evaluation_every',100,'evaluate the model after this many steps')
tf.flags.DEFINE_float('learning_rate',0.01,'the learning rate')
tf.flags.DEFINE_float('grad_clip',0.5,'the grad clip to prevent the grad explosion')


#其它参数
tf.flags.DEFINE_boolean('log_device_placement',False,'log placement of ops on devices')#是否打印设备分配的日志
tf.flags.DEFINE_boolean('allow_soft_placement',True,'allow TF soft placement')          #如果指定的设备不存在，允许TF自动分配设备


FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print ('all related parameters in :')
for attr,value in sorted(FLAGS.__flags.items()):
    print ('{}={}'.format(attr.upper(),value))


print ('参数打印完毕.....')


#加载数据
train_x,train_y,dev_x,dev_y=data_helper.load_dataset(FLAGS.raw_file)
print ('load data finished!')



with tf.Session() as sess:

    han=HAN_model.HAN(FLAGS.vocab_size,FLAGS.num_classes,FLAGS.embedding_size,FLAGS.hidden_size)


    with tf.name_scope('loss'):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=han.input_y,logits=han.out,name='loss'))

    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        label = tf.argmax(han.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())


    def train_step(x_batch, y_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 30,
            han.max_sentence_length: 30,
            han.batch_size: 64
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)

        time_str = str(int(time.time()))
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)

        return step


    def dev_step(x_batch, y_batch, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 30,
            han.max_sentence_length: 30,
            han.batch_size: 64
        }
        step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)
        time_str = str(int(time.time()))
        print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost,
                                                                                           accuracy))
        if writer:
            writer.add_summary(summaries, step)






    batches=data_helper.batch_iter(list(zip(train_x,train_y)),FLAGS.batch_size,FLAGS.num_epochs,shuffle=True)
    for batch in batches:
        x_batch,y_batch=zip(*batch)
        train_step(x_batch,y_batch)
        current_step=tf.train.global_step(sess,global_step)

        if current_step%FLAGS.evaluation_every==0:
            print('\nEvaluation!')
            dev_step(x_batch,y_batch)
            print ('\n')

        if current_step%FLAGS.checkpoint_every==0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
