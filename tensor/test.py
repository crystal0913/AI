import tensorflow as tf

# Reference:https://www.leiphone.com/news/201707/xSM8Q3Xx0JTmV0WY.html
# mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
# img = mnist.train.images[50]
# plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
# plt.show()
# print()

# with open('./saves/test.pkl', 'wb') as f:
# a=[6,7,8]
# pickle.dump(a, f)
# b = [6, 7, 8, 9]
# pickle.dump(b, f)
# pickle.dump([3,3,3], f)

# with open('./saves/test.pkl', 'rb') as f:
#     print(pickle.load(f))
#     print(pickle.load(f))
#     print(pickle.load(f))

# a = tf.random_normal([1, 3], stddev=1, seed=4)
# b = tf.random_normal([1, 3], stddev=1, seed=4)
# c = tf.random_normal([1, 3], stddev=1, seed=40)
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(b))
#     print(sess.run(c))

# def a():
#     print(b())
#
#
# def b():
#     return "x"
#
# a()
from tensorflow.contrib import rnn

units_num = 10  # 隐藏层节点

batch_size = 2  # 训练语料中一共有2句话
sequeue_len = 5  # 每句话只有5个词语
ebedding = 7  # 每个词语的词向量维度为 7

X = tf.random_normal(shape=[batch_size, sequeue_len, ebedding], dtype=tf.float32)


def cell():
    return rnn.BasicLSTMCell(units_num)  # 10代表10个节点


lstm_multi = rnn.MultiRNNCell([cell() for _ in range(2)], state_is_tuple=True)
state = lstm_multi.zero_state(batch_size, tf.float32)
output, state = tf.nn.dynamic_rnn(lstm_multi, X, initial_state=state, time_major=False)
finaloutput = output[-1]  # 取最后一个output
finaloutput_2 = output[: ,-1, :]
statec = state[-1].c  # 取最后一个state
stateh = state[-1].h
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # print('----------------output--------------------')
    # print(sess.run(output))  # 2*5*10
    print('----------------finaloutput--------------------')
    print(sess.run(finaloutput))  # 5*10
    print("----------")
    print(sess.run(finaloutput_2))  # 5*10
    # print('----------------state--------------------')
    # print(sess.run(state))  # 2*4*10
    # print('-----------------statec-------------------')
    # print(sess.run(statec))  # 2*10
    # print('----------------stateh--------------------')
    # print(sess.run(stateh))  # 2*10
# from tensorflow.python import debug as tfdbg
#
# labels = [[0.2, 0.3, 0.5],
#           [0.1, 0.6, 0.3]]
# logits = [[2, 0.5, 1],
#           [0.1, 1, 3]]
# logits_scaled = tf.nn.softmax(logits)
#
# result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# result2 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)
# result3 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# debug_sess = tfdbg.LocalCLIDebugWrapperSession(sess)  # dump_root="E:\\workspace\\py\\tensor\\a"
# print(debug_sess.run(result1))
# print(sess.run(result2))
# print(sess.run(result3))

# prepare the original data
# with tf.name_scope('data'):
#     x_data = np.random.rand(100).astype(np.float32)
#     y_data = 0.3 * x_data + 0.1
# # creat parameters
# with tf.name_scope('parameters'):
#     weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#     bias = tf.Variable(tf.zeros([1]))
# # get y_prediction
# with tf.name_scope('y_prediction'):
#     y_prediction = weight * x_data + bias
# # compute the loss
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.square(y_data - y_prediction))
# # creat optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# # creat train ,minimize the loss
# with tf.name_scope('train'):
#     train = optimizer.minimize(loss)
# # creat init
# with tf.name_scope('init'):
#     init = tf.global_variables_initializer()
# # creat a Session
# sess = tf.Session()
# # initialize
# writer = tf.summary.FileWriter("logs/", sess.graph)
# sess.run(init)
# # Loop
# for step in range(101):
#     sess.run(train)
#     if step % 10 == 0:
#         print(step, 'weight:', sess.run(weight), 'bias:', sess.run(bias))


# c = tf.Variable(initial_value=['a','b'])
# print(c.name)
# with tf.Session() as session:
#     a = tf.matmul([[3,4]], [[1],[2]])
#     print(a.eval())
#     session.close()
