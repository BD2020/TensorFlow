import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST processing using Deep Neural Network (Multilayer Perceptron Model)
#
# MNIST training + testing
#
# example flow of a feed forward neural network:
#
# input data > weight input data (with unique wts) > hidden layer 1 (activation function)
#   > weights > hidden layer 2 (activation function) > weights > output
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

# NN model: hidden layer 1, 2, 3
# node sizes don't need to be identical
#
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 # digits 0 - 9

# go through batches of 100 of our features and
# feed them our network one batch at a time, and
# manipulate the weights
#
batch_size = 100

# define a couple of place holder variables
#
# x is our input data with a specific shape: 784 pixels/values wide (28x28)
# x is height x width, but there is no height, just width 784
#
# y is the label of the data
#
# (if we don't define a shape here for x, and input shape doesn't match,
# TensorFlow will throw an error
#
x = tf.placeholder('float', [None, 784])

y = tf.placeholder('float')

# define our Deep Neural Network - Multilayer Perceptron Model
#
def neural_network_model(data):
    
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
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
  
    output_layer   = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
  

# Model: (input data * weights) + biases
# activation function (threshold: whether neuron fired)
# relu: rectilinear function is our threshold function
#
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

# our output layer
#
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output # returns one-hot array

# now we need to tell TensorFlow what to do with the NN model
# and session
#
# how we want to run data through the model
#

def train_neural_network(x):
    prediction = neural_network_model(x)

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

# define # of cycles of feedforward + back propagation
#
    num_epochs = 10 # use 10 for: 0.9487 accuracy, or: 20 for: 0.9568 accuracy
    
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

# with hidden layer # of nodes = 500:
#
# for num_epochs = 2:
# Epoch: 0 completed out of: 2 Loss: 1813321.25165
# Epoch: 1 completed out of: 2 Loss: 430404.777573
# Accuracy: 0.9168
# Done

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

# for num_epochs = 20:
# Epoch: 0 completed out of: 20 Loss: 1633100.55441
# # Epoch: 1 completed out of: 20 Loss: 398406.220016
# Epoch: 2 completed out of: 20 Loss: 221382.001685
# Epoch: 3 completed out of: 20 Loss: 132769.760822
# Epoch: 4 completed out of: 20 Loss: 82924.8462152
# Epoch: 5 completed out of: 20 Loss: 51485.1313419
# Epoch: 6 completed out of: 20 Loss: 35530.0142427
# Epoch: 7 completed out of: 20 Loss: 28782.3343085
# Epoch: 8 completed out of: 20 Loss: 21455.0283599
# Epoch: 9 completed out of: 20 Loss: 17786.5825066
# Epoch: 10 completed out of: 20 Loss: 17908.6600581
# Epoch: 11 completed out of: 20 Loss: 14498.6735606
# Epoch: 12 completed out of: 20 Loss: 15236.9442279
# Epoch: 13 completed out of: 20 Loss: 15899.0334228
# Epoch: 14 completed out of: 20 Loss: 13737.2685002
# Epoch: 15 completed out of: 20 Loss: 11597.5472009
# Epoch: 16 completed out of: 20 Loss: 12138.3050734
# Epoch: 17 completed out of: 20 Loss: 11491.5518536
# Epoch: 18 completed out of: 20 Loss: 12112.0146985
# Epoch: 19 completed out of: 20 Loss: 8453.81747318
# Accuracy: 0.9568

# with hidden layer # of nodes = 250:
#
# Epoch: 0 completed out of: 10 Loss: 794226.232082
# Epoch: 1 completed out of: 10 Loss: 190814.415005
# Epoch: 2 completed out of: 10 Loss: 114868.036423
# Epoch: 3 completed out of: 10 Loss: 75940.9888625
# Epoch: 4 completed out of: 10 Loss: 52594.2999949
# Epoch: 5 completed out of: 10 Loss: 37791.7335074
# Epoch: 6 completed out of: 10 Loss: 27371.2065346
# Epoch: 7 completed out of: 10 Loss: 19552.5692259
# Epoch: 8 completed out of: 10 Loss: 14072.7465534
# Epoch: 9 completed out of: 10 Loss: 11030.0385958
# Accuracy: 0.9358
# Done