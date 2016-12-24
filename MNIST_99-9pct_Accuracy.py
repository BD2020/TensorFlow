import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# from:
# https://www.tensorflow.org/tutorials/mnist/beginners/
#

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# input images x will consist of a 2d tensor of 
# floating point numbers. Here we assign it a shape 
# of [None, 784], where 784 is the dimensionality 
# of a single flattened 28 by 28 pixel MNIST image, 
# and None indicates that the first dimension, 
# corresponding to the batch size, can be of any size. 

# The shape argument to placeholder is optional, but 
# it allows TensorFlow to automatically catch bugs 
# stemming from inconsistent tensor shapes.

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# define the weights W and biases b for our model
#
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Before Variables can be used within a session, 
# they must be initialized using that session
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# implement our regression model. 
# We multiply the vectorized input images x 
# by the weight matrix W, add the bias b
#
y = tf.matmul(x,W) + b

# specify a loss function. Loss indicates how 
# bad the model's prediction was on a single example; 
# we try to minimize that while training across all
# the examples. Here, our loss function is the 
# cross-entropy between the target and the softmax 
# activation function applied to the model's prediction. 
#
# tf.nn.softmax_cross_entropy_with_logits internally 
# applies the softmax on the model's unnormalized 
# model prediction and sums across all classes, 
# and tf.reduce_mean takes the average over these sums
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# train the model:
# TensorFlow knows the entire computation graph, it 
# can use automatic differentiation to find the 
# gradients of the loss with respect to each of 
# the variables. TensorFlow has a variety of built-in 
# optimization algorithms. For this example, we will 
# use steepest gradient descent, with a step length 
# of 0.5, to descend the cross entropy.

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# The returned operation train_step, when run, will 
# apply the gradient descent updates to the parameters. 
# Training the model can therefore be accomplished by 
# repeatedly running train_step
#
for i in range(1000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})    
    
# how well did the model do?
#
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# jumping from a very simple model to something 
# moderately sophisticated: a small convolutional 
# neural network. This will get us to around 99.2% 
# accuracy -- not state of the art, but respectable

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Our convolutions uses a stride of one and are 
# zero padded so that the output is the same size 
# as the input. Our pooling is plain old max pooling 
# over 2x2 blocks. To keep our code cleaner, let's 
# also abstract those operations into functions.

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# First convolutiom layer:
#
# Implement our first layer. It will consist 
# of convolution, followed by max pooling. 
# The convolution will compute 32 features 
# for each 5x5 patch. Its weight tensor will 
# have a shape of [5, 5, 1, 32]
#
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 
# 4d tensor, with the second and third dimensions 
# corresponding to image width and height, and the 
# final dimension corresponding to the number of color channels
#
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x_image with the weight tensor, add 
# the bias, apply the ReLU function, and finally 
# max pool. The max_pool_2x2 method will reduce 
# the image size to 14x14.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Second Convolutional Layer
#
# In order to build a deep network, we stack 
# several layers of this type. The second layer 
# will have 64 features for each 5x5 patch.
#
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
#
# Now that the image size has been reduced 
# to 7x7, we add a fully-connected layer with 
# 1024 neurons to allow processing on the entire 
# image. We reshape the tensor from the pooling 
# layer into a batch of vectors, multiply by a 
# weight matrix, add a bias, and apply a ReLU
#
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
#
# To reduce overfitting, we will apply dropout 
# before the readout layer. We create a placeholder 
# for the probability that a neuron's output is 
# kept during dropout. This allows us to turn dropout 
# on during training, and turn it off during testing. 
# TensorFlow's tf.nn.dropout op automatically handles 
# scaling neuron outputs in addition to masking them, 
# so dropout just works without any additional scaling.1
#
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
#
# Finally, we add a layer, just like for the one layer 
# softmax regression above.
#
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
#
# How well does this model do? To train and 
# evaluate it we will use code that is nearly 
# identical to that for the simple one layer SoftMax network above.
#
# The differences are that:
#
# We will replace the steepest gradient descent optimizer 
#  with the more sophisticated ADAM optimizer.
#
# We will include the additional parameter keep_prob in 
#  feed_dict to control the dropout rate.
#
# We will add logging to every 100th iteration in the training process.
#

# it does 20,000 training iterations and may take 
# a while (possibly up to half an hour), depending on your processor.
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

# number of training iterations
#
num_train_epochs = 2000 # orig: 20000

for i in range(num_train_epochs):
    batch = mnist.train.next_batch(50)

    if i%100 == 0:
#        sess.run(train_accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_accuracy = accuracy.eval(session=sess,
                                       feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})

#        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("Step: ", i, "Training accuracy: ", train_accuracy)

        
    train_step.run(session=sess,
                   feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#print("Accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, 
#                                                           y_: mnist.test.labels, keep_prob: 1.0}))

print("Final accuracy: ", accuracy.eval(session=sess, feed_dict={x: mnist.test.images, 
                                                                 y_: mnist.test.labels, 
                                                                 keep_prob: 1.0}))


# print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# For this small convolutional network, performance
# is actually nearly identical with and without dropout. 
# Dropout is often very effective at reducing overfitting, 
# but it is most useful when training very large neural networks

# results:
# 
# Accuracy:  0.9196
# Step:  0 Training accuracy:  0.08
# Step:  100 Training accuracy:  0.78
# Step:  200 Training accuracy:  0.84
# Step:  300 Training accuracy:  0.94
# Step:  400 Training accuracy:  0.94
# Step:  500 Training accuracy:  0.92
# Step:  600 Training accuracy:  0.94
# Step:  700 Training accuracy:  1.0
# Step:  800 Training accuracy:  0.92
# Step:  900 Training accuracy:  0.96
# Step:  1000 Training accuracy:  0.9
# Step:  1100 Training accuracy:  1.0
# Step:  1200 Training accuracy:  0.98
# Step:  1300 Training accuracy:  0.94
# Step:  1400 Training accuracy:  0.96
# Step:  1500 Training accuracy:  0.94
# Step:  1600 Training accuracy:  0.92
# Step:  1700 Training accuracy:  1.0
# Step:  1800 Training accuracy:  0.98
# Step:  1900 Training accuracy:  1.0
