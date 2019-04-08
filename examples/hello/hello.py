# hello Tensorflow using Docker Jobber

import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')

with tf.Session() as sess:
    print(sess.run(hello))