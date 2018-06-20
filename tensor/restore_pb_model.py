from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# pb模型的恢复
def restore_model_pb():
    sess = tf.Session()
    tf.saved_model.loader.load(sess, ['mytag'], os.getcwd() + '\model2')
    input_x = sess.graph.get_tensor_by_name('input_x:0')
    op = sess.graph.get_tensor_by_name('predict:0')
    print(sess.run(op, feed_dict={input_x: np.expand_dims(mnist.test.images[15], axis=0)}))
    sess.close()


# ckpt模型的恢复
def restore_model_ckpt():
    sess = tf.Session()
    # 加载模型结构
    saver = tf.train.import_meta_graph('./save_path/file_name.meta')
    # 只需要指定目录就可以恢复所有变量信息
    saver.restore(sess, tf.train.latest_checkpoint('./save_path'))
    # 直接获取保存的变量
    print(sess.run('w:0'))

    input_x = sess.graph.get_tensor_by_name('input_x:0')
    # # 获取需要进行计算的operator
    op = sess.graph.get_tensor_by_name('predict:0')
    print(sess.run(op, feed_dict={input_x: np.expand_dims(mnist.test.images[15], axis=0)}))
    sess.close()

restore_model_pb()
# 打印所有变量的值
# print_tensors_in_checkpoint_file("save_path/file_name", None, True)
