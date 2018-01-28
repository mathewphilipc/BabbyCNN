import tensorflow as tf

# Placeholders work for simple experiments, but Datasets are the preferred method of
# streaming data into a model.


my_data = [
	[0,1],
	[2,3],
	[4,5],
	[6,7]
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_times = slices.make_one_shot_iterator().get_next()

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units = 1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
	_, loss_value = sess.run((train,loss))
	if (i%10 == 0):
		print(loss_value)