import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

# MNIST processing using RNN Recurrent Neural Network
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
# 

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

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

# no hidden layers with recurent neural networks
#
# define # of cycles of feedforward + back propagation
#
# hidden layers was:
# use 10 for: 0.9487 accuracy, or: 20 for: 0.9568 accuracy
#
num_epochs = 3   # was 10
    
num_classes = 10 # digits 0 - 9

# go through batches of 100 of our features and
# feed them our network one batch at a time, and
# manipulate the weights
#
batch_size = 128

chunk_size = 28
num_chunks = 28

# recurrent neural net size is 128
#
rnn_size = 128 # should be min of 128, or 512 etc

# define a couple of place holder variables
#
# y is the label of the data
#
# images are 28x28, so we'll do batches of this size
# 28 chunks of 28 pixels per chunk
#
x = tf.placeholder('float', [None, num_chunks, chunk_size])

y = tf.placeholder('float')

# define our RNN Recurrent Neural Network model
#
def recurrent_neural_network(x):
    
# weights: are a TF variable, that variable is a TF random normal
# then we specifiy shape of the weights: 784 x # nodes in hidden
# layer 1
#
# biases: are TF variables that are added in after the weights
#
# why are biases needed: if all the input data is 0, so no neuron would
# ever fire. So bias adds a value if all inputs were a zero
#
# the following will create a tensor or array of weights
# using data + random #'s
#
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,num_classes])),
             'biases': tf.Variable(tf.random_normal([num_classes]))}

# modify our input sata to be RNN ready
#
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, num_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    
# every cell will have outputs and states
#
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    
# our output layer
#
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output # returns one-hot array


# now we need to tell TensorFlow what to do with the NN model
# and session
#
# how we want to run data through the model
#

def train_neural_network(x):
    prediction = recurrent_neural_network(x)

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
    
# for RNN, we need to reshape epoch_x from 784
#
                epoch_x = epoch_x.reshape((batch_size, num_chunks, chunk_size))
    
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
# with RNN, we need to reshape our images
#
        print("Accuracy:", accuracy.eval({x:mnist.test.images.reshape((-1,num_chunks,chunk_size)), 
                                          y:mnist.test.labels}))

# train our neural network
#
train_neural_network(x)    

print("Done")

# Input MNIST data:
#
# Extracting /tmp/data\train-images-idx3-ubyte.gz
# Extracting /tmp/data\train-labels-idx1-ubyte.gz
# Extracting /tmp/data\t10k-images-idx3-ubyte.gz
# Extracting /tmp/data\t10k-labels-idx1-ubyte.gz
#

# with Deep Neural Network / hidden layers # of nodes = 500:
#
# for num_epochs = 2:
# Epoch: 0 completed out of: 2 Loss: 1813321.25165
# Epoch: 1 completed out of: 2 Loss: 430404.777573
# Accuracy: 0.9168
# Done

# with Deep Neural Network / hidden layers # of nodes = 500:
#
# for num_epochs = 10:
# Epoch: 0 completed out of: 10 Loss: 1785099.64235
# Epoch: 1 completed out of: 10 Loss: 414276.079529
# Epoch: 2 completed out of: 10 Loss: 224783.85342
# Epoch: 3 completed out of: 10 Loss: 133979.302543
# Epoch: 4 completed out of: 10 Loss: 79955.4939818
# Epoch: 5 completed out of: 10 Loss: 53943.2304412
# Epoch: 6 completed out of: 10 Loss: 37027.2919869
# Epoch: 7 completed out of: 10 Loss: 26976.376902
# Epoch: 8 completed out of: 10 Loss: 20682.3958718
# Epoch: 9 completed out of: 10 Loss: 19416.1112623
# Accuracy: 0.9487
# Done


# with RNN instead of Deep Neural Network with RNN size of 128
#
# for num_epochs: 3
#
# Epoch: 0 completed out of: 3 Loss: 185.841736279
# Epoch: 1 completed out of: 3 Loss: 51.797361739
# Epoch: 2 completed out of: 3 Loss: 34.8417065181
# Accuracy: 0.9769
# Done

# with RNN instead of Deep Neural Network with RNN size of 128
#
# for num_epochs: 10
#
# Epoch: 0 completed out of: 10 Loss: 201.06586222
# Epoch: 1 completed out of: 10 Loss: 52.501295004
# Epoch: 2 completed out of: 10 Loss: 36.2903537564
# Epoch: 3 completed out of: 10 Loss: 26.938566586
# Epoch: 4 completed out of: 10 Loss: 22.8711767318
# Epoch: 5 completed out of: 10 Loss: 18.7210224858
# Epoch: 6 completed out of: 10 Loss: 16.4110513597
# Epoch: 7 completed out of: 10 Loss: 14.43695963
# Epoch: 8 completed out of: 10 Loss: 12.56695476
# Epoch: 9 completed out of: 10 Loss: 12.1563876133
# Accuracy: 0.9789
# Done

# with RNN instead of Deep Neural Network with RNN size of 256
#
# for num_epochs: 3
#
# Epoch: 0 completed out of: 3 Loss: 195.687599011
# Epoch: 1 completed out of: 3 Loss: 48.5623333659
# Epoch: 2 completed out of: 3 Loss: 33.6198847163
# Accuracy: 0.9736
# Done  
