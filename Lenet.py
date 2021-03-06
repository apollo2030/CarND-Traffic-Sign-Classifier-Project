import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x26.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 7), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(7))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1_relu = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x26.
    conv1_maxpool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x160.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 7, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))

    conv2   = tf.nn.conv2d(conv1_maxpool, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2    = tf.nn.dropout(conv2, keep_prob)

    # SOLUTION: Activation.
    conv2_relu = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x160. Output = 5x5x160.
    conv2_maxpool = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x160.
    fc0   = flatten(conv2_maxpool)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 220.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*16, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Add dropout
    fc1    = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)


    # SOLUTION: Layer 4: Fully Connected. Input = 220. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

#     fc2    = tf.nn.dropout(fc2, dropout_probability)

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits, conv1, conv2, conv2_relu, conv2_maxpool, fc0, fc1, fc2