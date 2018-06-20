# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pickle

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

input_size = 784
hidden_size = 128
output_size = 10
epochs = 20000
batch_size = 50

input_tensor = tf.placeholder(tf.float32, [None, input_size], name='input')
output_tensor = tf.placeholder(tf.float32, [None, output_size], name='output')
w1 = tf.Variable(np.random.uniform(-1, 1, size=(input_size, hidden_size)), dtype=tf.float32, name='w_1')
w2 = tf.Variable(np.random.uniform(-1, 1, size=(hidden_size, output_size)), dtype=tf.float32, name='w_2')
b1 = tf.Variable(np.random.uniform(-1, 1, size=hidden_size), dtype=tf.float32, name='b_1')
b2 = tf.Variable(np.random.uniform(-1, 1, size=output_size), dtype=tf.float32, name='b_2')

a = tf.nn.softmax(tf.matmul(input_tensor, w1) + b1)
predict = tf.nn.softmax(tf.matmul(a, w2) + b2)
# loss_mse = tf.reduce_mean(tf.square(output_tensor - predict))
loss_mse = - tf.reduce_sum(output_tensor * tf.log(predict))
# loss_mse = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=output_tensor))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_mse)

correct_prediction = tf.equal(tf.argmax(output_tensor, 1), tf.argmax(predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
losses = []

saver = tf.train.Saver()


def train():
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(epochs):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={input_tensor: batch[0], output_tensor: batch[1]})
            if i % 50 == 0:
                ac = sess.run(accuracy, feed_dict={input_tensor: mnist.test.images,
                                                   output_tensor: mnist.test.labels})
                print(i, ":", ac)
                l = sess.run(loss_mse, feed_dict={input_tensor: batch[0], output_tensor: batch[1]})
                losses.append((ac * 100, l))
        # saver.save(sess, './saves/generator.ckpt')

    # 记录训练过程中的一些数据
    # with open('./saves/train_samples.pkl', 'wb') as f:
    #     pickle.dump(losses, f)


def draw_loss(losses):
    fig, ax = plt.subplots(figsize=(20, 7))
    losses = np.array(losses)
    plt.plot(losses.T[1], label='Loss')
    plt.plot(losses.T[0], label='accuracy')
    plt.title("Training Losses")
    plt.legend()
    plt.show()


def test():
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('saves'))
        test_sampes = [0, 10, 23, 46, 4039]
        for s in test_sampes:
            print(sess.run(predict, feed_dict={input_tensor: np.expand_dims(mnist.test.images[s], axis=0)}))
            print(sess.run(tf.argmax(sess.run(predict, feed_dict={
                input_tensor: np.expand_dims(mnist.test.images[s], axis=0)}), axis=1)))  # 预测结果
            print(mnist.test.labels[s])  # 标签值
            print('---------------------')

train()
# with open('./saves/train_samples.pkl', 'rb') as f:
#     losses = pickle.load(f)
# draw_loss(losses)

test()