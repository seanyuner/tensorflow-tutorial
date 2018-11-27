import tensorflow as tf

c = tf.constant("Hello from server1!")

# 生成一个有两个任务的集群，一个任务跑在本地2222端口，另外一个跑在本地2223端口。
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
# 通过上面生成的集群配置生成Server，并通过job_name和task_index指定当前所启动
# 的任务。因为该任务是第一个任务，所以task_index为0。
server = tf.train.Server(cluster, job_name="local", task_index=0)

sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) 
print(sess.run(c))
server.join()
