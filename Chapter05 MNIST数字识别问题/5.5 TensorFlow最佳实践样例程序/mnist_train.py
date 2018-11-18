import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# 加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"


def train(mnist):
    # 1. 定义神经网络的输入输出（网络参数由于mnist_inference函数，定义在了前向传播中）
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # 2. 定义前向传播、损失函数、反向传播
    # 前向传播（包含定义神经网络参数）
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    # 损失函数
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 反向传播（即优化算法）
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 3. 建立会话，训练
    saver = tf.train.Saver()   # 先初始化持久类
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 训练过程中，不再测试模型在验证数据上的表现，验证和测试的过程会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 这里仅输出模型在当前训练batch上的损失函数，大概来了解训练情况
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 保存当前模型，注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练轮数，如：model.ckpt-1000
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()