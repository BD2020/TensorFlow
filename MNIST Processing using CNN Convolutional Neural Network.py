import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST processing using Convolutional Neural Network CNN
#
# MNIST training + testing
#
# compare output to intended output > compare using a cost or loss function 
# (cross entropy)
#
# optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad, etc)
#
# TensorFlow has eight diff cost minimizers
#
# Minimizers go backwords and manipulates the weights to reduce cost (Back Propagation)
#
# feed forward + back propagation == epoch (one cycle)
#
# do this 10 - 20 times
# 
 
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# define # of cycles of feedforward + back propagation
#
num_epochs = 10

#output from out Neural Network NN will have say 10 nodes
#
# 10 digit classes (0 - 9)
#
# might say: 0 should output 0, 1 = 1, 2= 2, etc
#
# but one_hot = True will output:
# 
# 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#
# so only one element per output is on/hot, rest are off/cold
#

num_classes = 10 # digits 0 - 9

# go through batches of 100 of our features and
# feed them our network one batch at a time, and
# manipulate the weights
#
batch_size = 128

# define a couple of place hold variables
#
# x is our input data with a specific shape: 784 pixels/values wide (28x28)
# x is height x width, nut there is no height, just width 784
#
# y is the label of the data
#
# (if we don't define a shape here for x, and input shape doesn't match,
# TensorFlow will throw an error
#
x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, num_classes])

# convolution: for extracting features
# moving 1 pixel at a time ([1,1,1,1])
#
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# pooling for simplifying by finding max value in a feature
# take a 2x2 ksize window and pooling/moving it 2x2 pixels at a time
#
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
# define our CNN Convolutional Neural Network model
#
def convolutional_neural_network(x):
    
# weights: are a TF variable
#
# biases: are TF variables that are added in after the weights
#
# just need a weights and a biases dictionary
#
# why are biases needed: if all the input data is 0, so no neuron would
# ever fire. So bias adds a value if all inputs were a zero
#
# the following will create a tensor or array of weights
# using data + random #'s
#
# 5,5,1,32: 5x5 convolution, 1x input, and produce 32x features/outputs
#
# the fully connected/dense layer contains: 7*7 feature maps, not an image anymore
# along with 1024 nodes
#
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, num_classes]))} # classes: 0 - 9
    
    biases  = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([num_classes]))} # classes: 0 - 9

# reshape our input: from 784 to a 28x28 4D tensor
#
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

# taking 5x5 convolutions on the initial image, 
# and producing 32 outputs. Next, we take 5x5 
# convolutions of the 32 inputs and make 64 outputs. 
# From here, we're left with 7x7 sized images, 
# and 64 of them, and then we're outputting to 
# 1024 nodes in the fully connected layer. Then, 
# the output layer is 1024 layers, to 10, which 
# are the final 10 possible outputs for the actual 
# label itself (0-9).

# convolve then pass result for max pooling
# together these form one layer
#
# OLD
#    conv1 = conv2d(x, weights['W_conv1'])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    pool1 = maxpool2d(conv1)
    
# our 2nd convolution layer
# 
# OLD
#    conv2 = conv2d(conv1, weights['W_conv2'])
    conv2 = tf.nn.relu(conv2d(pool1, weights['W_conv2']) + biases['b_conv2'])
    pool2 = maxpool2d(conv2)
    
# next, compute our fully connected layer
#
    fc = tf.reshape(pool2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    
# next, out output layer
#
    output = tf.matmul(fc, weights['out']) + biases['out']
      
    return output

# now we need to tell TensorFlow what to do with the model
# and session
#
# how we want to run data through the model
#

def train_neural_network(x):
    prediction = convolutional_neural_network(x)

# now get our cost function
# calculates the difference between the prediction we got 
# and our known label
# output shape will always be the same as our testing set's labels
#
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )

# now we want to minimize the difference between the 
# prediction and y, ideally 0; so we use an optimizer
#
# AdamOptimizer has a default learning_rate of 0.001
#
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
#        sess.run(tf.initialize_all_variables())
        
        sess.run(tf.global_variables_initializer())

# train the model with our training data
#
        for epoch in range(num_epochs):
            epoch_loss = 0
        
# now loop through our batches
#
            for _ in range(int(mnist.train.num_examples / batch_size)):

# processes through the data in batch sizes
#
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) 

# get our cost
#
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})

                epoch_loss += c
        
            print("Epoch:", epoch, "completed out of:", num_epochs, "Loss:", epoch_loss)

# argmax returns the index of the max value of these arrays
#
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

# evaluate the accuracy of our test images to
# our test labels
#
        print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

# train our neural network
#
train_neural_network(x)    

print("Done")
