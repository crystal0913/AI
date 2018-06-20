# reference http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784], name='input_x')  # 输入图像占位符
y_ = tf.placeholder("float", shape=[None, 10])  # 标签类别占位符

# 模型参数一般用Variable来表示
W = tf.Variable(tf.zeros([784, 10]), name='w')  # 权重W是一个784x10的矩阵（因为我们有784个特征和10个输出值）
b = tf.Variable(tf.zeros([10]), name='b')  # 偏置b是一个10维的向量（因为我们有10个分类）

sess.run(tf.initialize_all_variables())  # 变量需要通过seesion初始化后，才能在session中使用
# 使用Tensorflow提供的回归模型softmax，y代表输出，把向量化后的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的softmax概率值
y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict')

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))  # 计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 梯度下降算法以0.01的学习速率最小化交叉熵

# tf.argmax返回某个tensor对象在某一维上的其数据最大值所在的索引值
# 下面这行返回一组布尔值如[True, False, True, True]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 把布尔值转换成浮点数，然后取平均值,[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
loss = []
for i in range(1000):
    batch = mnist.train.next_batch(50)  # 每一步迭代加载50个训练样本，然后执行一次train_step
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 模型在测试数据集上面的正确率
        # cross_entropy_1 = sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1]})
        # print(sess.run(W[8]))
        # loss.append(sess.run([W[8][8], b[3]]))

print(sess.run(b))
w_ = sess.run(W)
for i in range(w_.shape[0]):
    print(w_[i])
# print(loss)


def draw_loss(losses):
    fig, ax = plt.subplots(figsize=(20, 7))
    losses = np.array(losses)
    plt.plot(losses.T[0], label='w')
    plt.plot(losses.T[1], label='b')
    plt.title("Training Losses")
    plt.legend()
    plt.show()
# draw_loss(loss)

# 随便从测试集中取一个例子做测试
print(sess.run(y, feed_dict={x: np.expand_dims(mnist.test.images[15], axis=0)}))
print(sess.run(tf.argmax(sess.run(y, feed_dict={x: np.expand_dims(mnist.test.images[15], axis=0)}), axis=1)))   # 预测结果
print(mnist.test.labels[15])    # 标签值

# pb模型的保存
# builder = tf.saved_model.builder.SavedModelBuilder('./model2')
# builder.add_meta_graph_and_variables(sess, ["mytag"])
# builder.save()

# ckpt模型的保存
# saver = tf.train.Saver()
# saver.save(sess, "save_path/file_name")  # file_name.ckpt如果不存在的话，会自动创建

sess.close()


