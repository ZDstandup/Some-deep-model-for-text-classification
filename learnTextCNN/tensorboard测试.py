#__author__=="zhangdong"
#@zhangdong

import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
sess.run(e)

writer = tf.summary.FileWriter("E:/Mr ZhangDong/Zhangdong_NLPDeep/textCNN_zd/log", tf.get_default_graph())
writer.close()

