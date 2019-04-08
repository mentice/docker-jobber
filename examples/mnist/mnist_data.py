import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

mnist_data = mnist.load_data()

np.save('/data/mnist.npy', mnist_data)
