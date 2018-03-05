# https://github.com/aymericdamien/TensorFlow-Examples/ (cont.)
# blob/master/examples/5_DataManagement/build_an_image_dataset.py

from __future__ import print_function

import tensorflow as tf
import numpy as np

import tensorflow as tf
import os
import numpy as np


MINI_OR_FULL = "MINI"
MINI_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45-MINI"
FULL_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45"


MODE = 'folder'
DATASET_PATH = FULL_DATASET_PATH

MINI_N_CLASSES = 10
FULL_N_CLASSES = 45
N_CLASSES = FULL_N_CLASSES

# print(N_CLASSES)

IMG_HEIGHT = 32 # original size = 256
IMG_WIDTH = 32 # original size = 256
CHANNELS = 3 # we have full-color images


TRAIN_FRAC = 0.90


# Read dataset
def read_images(dataset_path, mode, batch_size):
    train_imagepaths = list()
    test_imagepaths = list()
    train_labels = list()
    test_labels = list()
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            train_imagepaths.append(d.split(' ')[0])
            test_imagepaths.append(d.split(' ')[0])
            train_labels.append(int(d.split(' ')[1]))
            test_labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # Count how many (image, label) pairs go into testing vs training
        total_test_count = 0;
        total_train_count = 0;
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        image_count = 0;
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            #print("Now on class number %d: %s" % (label, c))
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    test_or_train = np.random.rand()
                    # Add image+label to either test or train set
                    if (test_or_train > TRAIN_FRAC):
                        test_imagepaths.append(os.path.join(c_dir, sample))
                        test_labels.append(label)
                        total_test_count += 1

                    else:
                        train_imagepaths.append(os.path.join(c_dir, sample))
                        train_labels.append(label)
                        total_train_count += 1
                    image_count += 1
            label += 1
            #print("Just added image number %d" % label)
    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    train_imagepaths = tf.convert_to_tensor(train_imagepaths, dtype=tf.string)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
    test_imagepaths = tf.convert_to_tensor(test_imagepaths, dtype=tf.string)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    train_image, train_label = tf.train.slice_input_producer([train_imagepaths, train_labels],
                                                 shuffle=True)
    test_image, test_label = tf.train.slice_input_producer([test_imagepaths, test_labels],
                                                 shuffle=True)

    # Read images from disk
    train_image = tf.read_file(train_image)
    train_image = tf.image.decode_jpeg(train_image, channels=CHANNELS)
    test_image = tf.read_file(test_image)
    test_image = tf.image.decode_jpeg(test_image, channels=CHANNELS)

    # Resize images to a common size
    train_image = tf.image.resize_images(train_image, [IMG_HEIGHT, IMG_WIDTH])
    test_image = tf.image.resize_images(test_image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    train_image = train_image * 1.0/127.5 - 1.0
    test_image = test_image * 1.0/127.5 - 1.0

    # Create batches

    print("\nFound all images and labels in NWPU-RESISC45...\n")

    return train_image, test_image, train_label, test_label, total_train_count, total_test_count
















# Set hyperparameters

learning_rate = 0.000001
num_steps = 10000
batch_size = 1000
display_step = 1
dropout = 0.4

# Build the data input
#X_train, Y_train = read_images(DATASET_PATH, MODE, batch_size)

train_image, test_image, train_label, test_label, total_train_count, total_test_count = read_images(DATASET_PATH, MODE, batch_size)

X_train, Y_train = tf.train.batch([train_image, train_label], batch_size=batch_size,
    capacity=batch_size * 8, num_threads=4)

# Use entire testing set for every accuracy check
X_test, Y_test = tf.train.batch([test_image, test_label], batch_size=total_test_count,
    capacity=batch_size * 8, num_threads=4)


print("\nDone randomly selecting %d training images and %d test images\n" % (total_train_count, total_test_count))

N_DIGITS = FULL_N_CLASSES


def incept_layer(inputlayer, filters1a, filters1b, filters2a, filters2b, filters2c, filters2d):
        layer1a = tf.layers.conv2d(
            inputs = inputlayer,
            filters = filters1a,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        layer1b = tf.layers.conv2d(
            inputs = inputlayer,
            filters = filters1a,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        layer1c = tf.layers.max_pooling2d(
            inputs = inputlayer,
            pool_size = 3,
            strides = 1,
            padding="same")

        layer2a = tf.layers.conv2d(
            inputs = inputlayer,
            filters = filters2a,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        layer2b = tf.layers.conv2d(
            inputs = layer1a,
            filters = filters2b,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        layer2c = tf.layers.conv2d(
            inputs = layer1a,
            filters = filters2c,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        layer2d = tf.layers.conv2d(
            inputs = layer1a,
            filters = filters2d,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        layerOut = tf.concat([layer2a, layer2b, layer2c, layer2d], 3)

        return layerOut





def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):

        conv1 = tf.layers.conv2d(
            inputs = x,
            filters = 64,
            kernel_size = 7,
            strides = 2,
            padding = "same",
            activation=tf.tanh)

        pool2 = tf.layers.max_pooling2d(
            inputs = conv1,
            pool_size = 3,
            strides = 2,
            padding="same")

        norm3 = tf.nn.local_response_normalization(
            input = pool2,
            depth_radius = 4,
            bias=1.0,
            alpha = 1.0,
            beta = 0.5,
            name = None)

        conv4 = tf.layers.conv2d(
            inputs = norm3,
            filters = 64,
            kernel_size = 1,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        conv5 = tf.layers.conv2d(
            inputs = conv4,
            filters = 192,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.tanh)

        norm6 = tf.nn.local_response_normalization(
            input = conv5,
            depth_radius = 4,
            bias=1.0,
            alpha = 1.0,
            beta = 0.5,
            name = None)

        pool7 = tf.layers.max_pooling2d(
            inputs = norm6,
            pool_size = 3,
            strides = 2,
            padding="same")


        # First inception module



# inputlayer, filters1a, filters1b, filters1c, filters2a, filters2b, filters2c, filters2d

        incept8 = incept_layer(
            inputlayer = pool7,
            filters1a = 96,
            filters1b = 16, 
            filters2a = 64, 
            filters2b = 128,
            filters2c = 32,
            filters2d = 32)

        incept9 = incept_layer(incept8, 128, 32, 128, 192, 96, 64)

        pool10 = tf.layers.max_pooling2d(
            inputs = incept9,
            pool_size = 3,
            strides = 2,
            padding="same")

        incept11 = incept_layer(pool10, 96, 16, 192, 208, 48, 64)
        incept12 = incept_layer(incept11, 112, 24, 160, 224, 64, 64)
        incept13 = incept_layer(incept12, 128, 24, 128, 256, 64, 64)
        incept14 = incept_layer(incept13, 144, 32, 112, 288, 64, 64)
        incept15 = incept_layer(incept14, 160, 32, 256, 320, 128, 128)

        pool16 = tf.layers.max_pooling2d(
            inputs = incept15,
            pool_size = 3,
            strides = 2,
            padding = "same")

        incept17 = incept_layer(pool16, 160, 32, 256, 320, 128, 128)
        incept18 = incept_layer(incept17, 192, 48, 384, 384, 128, 128)

        finalPool = tf.layers.average_pooling2d(
            inputs = incept18,
            pool_size = 7,
            strides = 1,
            padding = "same")

        flattened = tf.contrib.layers.flatten(finalPool)

        connected = tf.layers.dense(
            inputs = flattened,
            units = 1000)

        out = tf.layers.dense(connected, n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out

logits_train = conv_net(X_train, N_CLASSES, dropout, reuse=False, is_training=True)
logits_test = conv_net(X_test, N_CLASSES, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_test_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_test, tf.int64))
test_accuracy = tf.reduce_mean(tf.cast(correct_test_pred, tf.float32))

correct_train_pred = tf.equal(tf.argmax(logits_train, 1), tf.cast(Y_train, tf.int64))
train_accuracy = tf.reduce_mean(tf.cast(correct_train_pred, tf.float32))




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
            _, loss, test_acc, train_acc = sess.run([train_op, loss_op, test_accuracy, train_accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Train Acc " + \
                  "{:.3f}".format(train_acc) + ", Test Acc = " + \
                  "{:.3f}".format(test_acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")
