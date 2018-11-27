import tensorflow as tf

# 在CPU上运行tf.Variable
a_cpu = tf.Variable(0, name="a_cpu")

with tf.device('/gpu:0'):
    # 将tf.Variable强制放在GPU上
    a_gpu = tf.Variable(0, name="a_gpu")
    
# 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())
