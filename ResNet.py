# https://github.com/aymericdamien/TensorFlow-Examples/ (cont.)
# blob/master/examples/5_DataManagement/build_an_image_dataset.py

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np

# Toggle this to False if you're continuing from previous training
FIRST_TRAINING_SESSION = True

MODEL_PATH = "/home/mathew/NWPU_Models/ResNet/"
#MODEL_PATH = "/home/ubuntu/NWPU_Models/ResNet/"
MINI_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45-MINI"
#MINI_DATASET_PATH = "/home/ubuntu/data/NWPU-RESISC45-MINI"
FULL_DATASET_PATH = "/home/mathew/Desktop/NWPU-RESISC45"
#FULL_DATASET_PATH = "/home/ubuntu/data/NWPU-RESISC45"


MODE = 'folder'
DATASET_PATH = FULL_DATASET_PATH

MINI_N_CLASSES = 10
FULL_N_CLASSES = 45
N_CLASSES = FULL_N_CLASSES

# print(N_CLASSES)

IMG_HEIGHT = 64 # original size = 256
IMG_WIDTH = 64 # original size = 256
CHANNELS = 3 # we have full-color images


TRAIN_FRAC = 0.95

# For deterministic, consistent train/test splitting across runs
np.random.seed(0)


# Read dataset
def read_images(dataset_path, mode, batch_size):
    train_imagepaths = list()
    test_imagepaths = list()
    train_labels = list()
    test_labels = list()
    # Count how many (image, label) pairs go into testing vs training
    [total_test_count, total_train_count] = [0, 0]
    [label, image_count] = [0, 0];
    classes = sorted(os.walk(dataset_path).next()[1])
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).next()
        # Add each image to the training set
        for sample in walk[2]:
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

    # Read, resize, and normalize
    train_image = tf.image.decode_jpeg(tf.read_file(train_image), channels=CHANNELS)
    train_image = tf.image.resize_images(train_image, [IMG_HEIGHT, IMG_WIDTH]) * 1.0/127.5 - 1.0
    test_image = tf.image.decode_jpeg(tf.read_file(test_image), channels=CHANNELS)
    test_image = tf.image.resize_images(test_image, [IMG_HEIGHT, IMG_WIDTH]) * 1.0/127.5 - 1.0


    print("\nFound all images and labels in NWPU-RESISC45...\n")

    return train_image, test_image, train_label, test_label, total_train_count, total_test_count


# Set hyperparameters

learning_rate = 0.0003
num_steps = 5
batch_size = 1000
display_step = 1
dropout = 0.0

# Build the data input

train_image, test_image, train_label, test_label, total_train_count, total_test_count = read_images(DATASET_PATH, MODE, batch_size)

test_batch_size = total_test_count // 10




X_train, Y_train = tf.train.batch([train_image, train_label], batch_size=batch_size,
    capacity=batch_size * 8, num_threads=4)

# Use entire testing set for every accuracy check
X_test, Y_test = tf.train.batch([test_image, test_label], batch_size=test_batch_size,
    capacity=batch_size * 8, num_threads=4)

print("\nDone randomly selecting %d training images and %d test images\n" % (total_train_count, total_test_count))

N_DIGITS = FULL_N_CLASSES



def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):

        # same start as GoogLeNet
        initConv = tf.layers.conv2d(
            inputs = x,
            filters = 64,
            kernel_size = 7,
            strides = 2,
            padding = "same",
            activation=tf.tanh)

        initPool = tf.layers.max_pooling2d(
            inputs = initConv,
            pool_size = 3,
            strides = 2,
            padding="same")

        # first residual block
        res1a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = initPool,
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.nn.relu))
        res1b = tf.layers.conv2d(
            inputs = res1a,
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None)
        res1 = tf.nn.relu(initPool + tf.contrib.layers.batch_norm(inputs = res1b))





        res2a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res1,
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.nn.relu))
        res2b = tf.contrib.layers.batch_norm(tf.layers.conv2d(
            inputs = res2a,
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res2 = tf.nn.relu(res1 + res2b)





        res3a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res2,
            filters = 128,
            kernel_size = 3,
            strides = 2,
            padding = "same",
            activation=tf.nn.relu))
        res3b = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res3a,
            filters = 128,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res3c = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res2,
            filters = 128,
            kernel_size = 1,
            strides = 2,
            padding="same",
            activation = None))
        res3 = tf.nn.relu(res3c + res3b)





        res4a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res3,
            filters = 128,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.nn.relu))
        res4b = tf.contrib.layers.batch_norm(tf.layers.conv2d(
            inputs = res4a,
            filters = 128,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res4 = tf.nn.relu(res3 + res4b)





        res5a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res4,
            filters = 256,
            kernel_size = 3,
            strides = 2,
            padding = "same",
            activation=tf.nn.relu))
        res5b = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res5a,
            filters = 256,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res5c = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res4,
            filters = 256,
            kernel_size = 1,
            strides = 2,
            padding="same",
            activation = None))
        res5 = tf.nn.relu(res5c + res5b)

        # res block 6
        res6a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res5,
            filters = 256,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.nn.relu))
        res6b = tf.contrib.layers.batch_norm(tf.layers.conv2d(
            inputs = res6a,
            filters = 256,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res6 = tf.nn.relu(res5 + res6b)





        res7a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res6,
            filters = 512,
            kernel_size = 3,
            strides = 2,
            padding = "same",
            activation=tf.nn.relu))
        res7b = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res7a,
            filters = 512,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res7c = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res6,
            filters = 512,
            kernel_size = 1,
            strides = 2,
            padding="same",
            activation = None))
        res7 = tf.nn.relu(res7c + res7b)





        res8a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res7,
            filters = 512,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.nn.relu))
        res8b = tf.contrib.layers.batch_norm(tf.layers.conv2d(
            inputs = res8a,
            filters = 512,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res8 = tf.nn.relu(res7 + res8b)





        res9a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res8,
            filters = 1024,
            kernel_size = 3,
            strides = 2,
            padding = "same",
            activation=tf.nn.relu))
        res9b = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res9a,
            filters = 1024,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res9c = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res8,
            filters = 1024,
            kernel_size = 1,
            strides = 2,
            padding="same",
            activation = None))
        res9 = tf.nn.relu(res9c + res9b)





        res10a = tf.contrib.layers.batch_norm(inputs = tf.layers.conv2d(
            inputs = res9,
            filters = 1024,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation=tf.nn.relu))
        res10b = tf.contrib.layers.batch_norm(tf.layers.conv2d(
            inputs = res10a,
            filters = 1024,
            kernel_size = 3,
            strides = 1,
            padding = "same",
            activation = None))
        res10 = tf.nn.relu(res9 + res10b)

        # done with res blocks
        # Time for avgpool, then a dense layer (sans dropout), then softmax

        finalPool = tf.layers.average_pooling2d(
            inputs = res10,
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


topfive_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_test, Y_test, 5), tf.float32))
#topthree_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_test, Y_test, 3), tf.float32))




init = tf.global_variables_initializer()

saver = tf.train.Saver()


if not FIRST_TRAINING_SESSION:
    imported_meta = tf.train.import_meta_graph(MODEL_PATH + "model.ckpt.meta")


with tf.Session() as sess:

    if FIRST_TRAINING_SESSION:
        sess.run(init)
    else:
        imported_meta.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))



    #Start the data queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, test_acc, topfive_acc = sess.run([train_op, loss_op, test_accuracy, topfive_accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Test Acc " + \
                  "{:.3f}".format(test_acc) + ", Top-5 Test Acc = " + \
                  "{:.3f}".format(topfive_acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)
    coord.request_stop()
    coord.join(threads)

    print("Optimization Finished!")

    # Save model

    save_path = saver.save(sess, MODEL_PATH + "model.ckpt")
    print("Model saved in path: %s" % save_path)