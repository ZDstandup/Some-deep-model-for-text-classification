#coding=utf-8
#author@zhangdong
import datetime
import tensorflow as tf
import data_helper
from tensorflow.contrib import learn
import numpy as np
from TextRCNN_model import TextRCNN
import os
import time




#configuration
FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('raw_train_file','./../data/ag_news_csv/train.csv','the raw training file')
tf.flags.DEFINE_string('raw_test_file','./../data/ag_news_csv/test.csv','the raw testing file')
tf.flags.DEFINE_integer("num_classes",4,"number of label")
tf.flags.DEFINE_float('dev_percent',0.1,'the percent of the train data to evaluate')
tf.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.flags.DEFINE_string("ckpt_dir","text_rcnn_title_desc_checkpoint2/","checkpoint location for the model")
tf.flags.DEFINE_integer("sequence_length",198,"max sentence length")
tf.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.flags.DEFINE_integer("num_epochs",60,"number of epochs to run.")
tf.flags.DEFINE_integer("evaluation_every", 100, "Validate every validate_every epochs.") #每10轮做一次验证
tf.flags.DEFINE_integer('checkpoint_every',100,'save the model after this many steps default:100')
tf.flags.DEFINE_integer('num_checkpoints',5,'num of model saving')
tf.flags.DEFINE_integer('dropout_keep_prob',0.5,'the dropout prob')
tf.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")




#load data
train_x_text,train_y=data_helper.load_dataset(FLAGS.raw_train_file)
test_x_text,test_y=data_helper.load_dataset(FLAGS.raw_test_file)
all_x_text=train_x_text+test_x_text

#build vocabulary
max_document_length=max([len(x.split(' ')) for x in all_x_text])        #198有点长
vocab_processor=learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length,min_frequency=3)

train_x=np.array(list(vocab_processor.fit_transform(train_x_text)))
text_x=np.array(list(vocab_processor.fit_transform(test_x_text)))

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print ('all related parameters in RCNN:')
for attr,value in sorted(FLAGS.__flags.items()):
    print ('{}={}'.format(attr.upper(),value))


print ('参数打印完毕.....')

print (max_document_length)
vocab_size=len(vocab_processor.vocabulary_)
print (vocab_size)
print (train_x.shape)
print (train_x[0])



#randomly shuffle the train data
np.random.seed(10)
shuffle_indices=np.random.permutation(len(train_y))
train_x_shuffled=train_x[shuffle_indices]
train_y_shuffled=np.array(train_y)[shuffle_indices]


dev_index=-1*(int(len(train_y)*FLAGS.dev_percent))
x_train,y_train=train_x_shuffled[:dev_index],train_y_shuffled[:dev_index]
x_dev,y_dev=train_x_shuffled[dev_index:],train_y_shuffled[dev_index:]




# print(train_x.shape)
# print (train_y)
# print (len(y_dev))
# print (x_dev.shape)
# print (y_dev.shape)

print ('train/dev : {}/{}'.format(len(y_train),len(y_dev)))
print ('start to RCNN training ......')

#print(y_train[:10])



with tf.Graph().as_default():
    session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess=tf.Session(config=session_config)
    with sess.as_default():
        textRCNN = TextRCNN(num_classes=FLAGS.num_classes, learning_rate=FLAGS.learning_rate,
                                  batch_size=FLAGS.batch_size,
                                  decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
                                  sequence_length=FLAGS.sequence_length,
                                  vocab_size=vocab_size, is_training=FLAGS.is_training,
                                    embed_size=FLAGS.embed_size,
                                  initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False)



        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        # 存储模型
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        def train_step(x_batch,y_batch):

            loss, acc, predict, _ ,step= sess.run([textRCNN.loss_val, textRCNN.accuracy, textRCNN.predictions, textRCNN.train_op,textRCNN.global_step],
                                         feed_dict={textRCNN.input_x: x_batch,
                                                    textRCNN.input_y: y_batch,
                                                    textRCNN.dropout_keep_prob: FLAGS.dropout_keep_prob})

            time_str = datetime.datetime.now().isoformat()

            print("{}: step {:d}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            #print("loss:", loss, "acc:", acc, "label:", y_batch, "prediction:", predict)

        def dev_step(x_batch,y_batch):

            num_dev=len(x_batch)
            count=0
            all_acc=0.0;all_loss=0.0
            for start, end in zip(range(0, num_dev, FLAGS.batch_size),
                                  range(FLAGS.batch_size, num_dev + 1, FLAGS.batch_size)):
                dev_x_batch,dev_y_batch=x_batch[start:end],y_batch[start:end]
                count+=1

                loss, acc, predict, step= sess.run([textRCNN.loss_val, textRCNN.accuracy, textRCNN.predictions, textRCNN.global_step],
                                             feed_dict={textRCNN.input_x: dev_x_batch,
                                                        textRCNN.input_y: dev_y_batch,
                                                        textRCNN.dropout_keep_prob: 1.0})
                all_acc+=acc
                all_loss+=loss


            dev_loss=all_loss/count
            dev_acc=all_acc/count
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {:d}, loss {:g}, acc {:g}".format(time_str, step, dev_loss, dev_acc))


        sess.run(tf.global_variables_initializer())

        batches = data_helper.batch_iter(data=list(zip(x_train, y_train)), batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, shuffle=True)
        for batch in batches:  # [(x1,y1),(x2,y2),...,(x64,y64)]
            x_batch_, y_batch_ = zip(*batch)  # [x1,x2,...x64]  [y1,y2,...,y64]
            train_step(x_batch_, y_batch_)
            current_step = tf.train.global_step(sess, textRCNN.global_step)

            if current_step % FLAGS.evaluation_every == 0:
                print("\nEvaluation....")
                dev_step(x_dev, y_dev)
                print('\n')

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))