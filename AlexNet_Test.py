# https://github.com/aymericdamien/TensorFlow-Examples/ (cont.)
# blob/master/examples/5_DataManagement/build_an_image_dataset.py

from __future__ import print_function

import tensorflow as tf
import os

MINI_OR_FULL = "MINI"
MINI_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45-MINI"
FULL_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45"


MODE = 'folder'
DATASET_PATH = FULL_DATASET_PATH

MINI_N_CLASSES = 10
FULL_N_CLASSES = 45
N_CLASSES = FULL_N_CLASSES

print(N_CLASSES)

IMG_HEIGHT = 224 # original size = 256
IMG_WIDTH = 224 # original size = 256
CHANNELS = 3 # we have full-color images





















# Read dataset
def read_images(dataset_path, mode, batch_size):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y






















# Set hyperparameters

learning_rate = 0.001
num_steps = 1000
batch_size = 100
display_step = 1
dropout = 0.5

# Build the data input
X, Y = read_images(DATASET_PATH, MODE, batch_size)

def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        conv1 = tf.layers.conv2d(
            inputs = x,
            filters = 96, # previous: filters = 32
            kernel_size = [11, 11],
            strides = (4,4),
            padding = "same",
            activation=tf.nn.relu)
            # Input Tensor Shape: [batch_size, 32, 32, 1]
            # Output Tensor Shape: [batch_size, 32, 32, 96]

        pool2 = tf.layers.max_pooling2d(
            inputs = conv1,
            pool_size = 3,
            strides = 2)
            # Input Tensor Shape: [batch_size, 32, 32, 32]
            # Output Tensor Shape: [batch_size, 16, 16, 32]

        conv3 = tf.layers.conv2d(
        inputs = pool2,
        filters = 256,
        kernel_size = [5, 5],
        padding="same",
        activation=tf.nn.relu)
        # Input Tensor Shape: [batch_size, 16, 16, 32]
        # Output Tensor Shape: [batch_size, 16, 16, 64]


        pool4 = tf.layers.max_pooling2d(
            inputs = conv3,
            pool_size = [3, 3],
            strides = 2)
        # ;lkajsdf

        conv5 = tf.layers.conv2d(
            inputs = pool4,
            filters = 384,
            kernel_size = 3,
            padding = "same",
            activation = tf.nn.relu)

        conv6 = tf.layers.conv2d(
            inputs = conv5,
            filters = 384,
            kernel_size = 3,
            padding="same",
            activation = tf.nn.relu)

        conv7 = tf.layers.conv2d(
            inputs = conv6,
            filters = 384,
            kernel_size = 3,
            padding="same",
            activation = tf.nn.relu)


        flattened = tf.contrib.layers.flatten(conv7)

        fc1 = tf.layers.dense(flattened, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out

logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")

    # Save your model
    # saver.save(sess, 'my_tf_model')
    saver.save(sess, '/home/mathew/AlexNet_model/AlexNet_model')