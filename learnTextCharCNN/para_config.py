#coding=utf-8
#author@zhangdong

#把本次实验的关于模型，训练等参数写在config文件中，并没有在train.py中进行相关的输出。其实也可以输出，这样代码更加具有易读性


class TrainingConfig(object):
    decay_step=15000
    decay_rate=0.95
    epoches=50000
    evaluate_every=100
    checkpoint_every=100



class ModelConfig(object):
    conv_layers=[[256, 7, 3],
                 [256, 7, 3],
                 [256, 3, None],
                 [256, 3, None],
                 [256, 3, None],
                 [256, 3, 3]]

    fc_layers=[1024,1024]
    dropout_keep_prob=0.9
    learning_rate=0.001


class Config(object):
    alphalet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphalet_size=len(alphalet)
    l0=1014     #每条文本最多1014个字符
    batch_size=128
    num_classes=4
    example_nums=120000         #？？？？

    train_data_source = './../data/ag_news_csv/train.csv'
    dev_data_source = './../data/ag_news_csv/test.csv'

    training=TrainingConfig()
    model=ModelConfig()


config=Config()

# print (config.training.checkpoint_every)

