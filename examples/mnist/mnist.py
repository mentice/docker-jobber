import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

(x_train, y_train),(x_test, y_test) = np.load('/data/mnist.npy')
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation=tf.nn.relu),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)
loss, metrics = model.evaluate(x_test, y_test)
print('loss={0} accuracy={1}'.format(loss, metrics))