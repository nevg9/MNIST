
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().magic(u'matplotlib inline')


# In[ ]:


mnist = input_data.read_data_sets("MNIST_inputdata/", one_hot=True)


# In[ ]:


x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[ ]:


y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# In[ ]:


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# In[ ]:


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# In[ ]:


x_t , y_t = mnist.train.next_batch(1)
print(x_t.shape)


# # MNIST 进阶
# ## lenet-5实现
# **tensorflow 的实现步骤：**
# 
# 1. create_placeholders
# 2. initialize parameters
# 3. forward propagate
# 4. compute the cost
# 5. create an optimizer
# 

# In[ ]:


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    x = tf.placeholder("float", shape=[None, n_H0, n_W0, n_C0])
    y_ = tf.placeholder("float", shape=[None, n_y])
    return x, y_


# In[ ]:


def initialize_parameters():
    tf.set_random_seed(1)                              
        
    W1 = tf.get_variable("W1", [5, 5, 1, 6], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    b1 =  tf.get_variable("b1", [6], initializer=tf.constant_initializer(0.0))
    W2 = tf.get_variable("W2", [5, 5, 6, 16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    b2 =  tf.get_variable("b2", [16], initializer=tf.constant_initializer(0.0))
    W3 = tf.get_variable("W3", [5*5*16, 120], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    b3 =  tf.get_variable("b3", [120], initializer=tf.constant_initializer(0.0))
    W4 = tf.get_variable("W4", [120, 84], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    b4 =  tf.get_variable("b4", [84], initializer=tf.constant_initializer(0.0))
    W5 = tf.get_variable("W5", [84, 10], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    b5 =  tf.get_variable("b5", [10], initializer=tf.constant_initializer(0.0))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}
    
    return parameters


# In[ ]:


def forward_propagation(X, parameters, keep_prob):
    """
    
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    print(X.shape)
    Z1 = tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(tf.nn.bias_add(Z1, b1))
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, [1, 1, 1, 1], padding='VALID')
    # RELU
    A2 = tf.nn.relu(tf.nn.bias_add(Z2, b2))
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    A3 = tf.nn.relu(tf.matmul(P2, W3) + b3)
    A3_drop = tf.nn.dropout(A3, keep_prob)
    A4 = tf.nn.relu(tf.matmul(A3_drop,W4) + b4)
    A4_drop = tf.nn.dropout(A4, keep_prob)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z5 = tf.matmul(A4_drop, W5) + b5
    ### END CODE HERE ###

    return Z5


# In[ ]:


def compute_cost(Z5, Y):
    """
    Computes the cost
    
    Arguments:
    Z5 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z5
    
    Returns:
    cost - Tensor of the cost function
    """
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y))
    ### END CODE HERE ###
    
    return cost


# In[ ]:


def model(mnist, learning_rate = 0.009,
          num_epochs = 1000, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                 # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                     # to keep results consistent (tensorflow seed)
    seed = 3                                  # to keep results consistent (numpy seed)
    n_H0, n_W0, n_C0 = (28, 28, 1)
    n_y = 10                         
    costs = []                                # To keep track of the cost
    
    # Create Placeholders of the correct shape
   
    x, y_ = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    keep_prob = tf.placeholder("float")
    # Initialize parameters
   
    parameters = initialize_parameters()
   
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    
    Z5 = forward_propagation(x, parameters, keep_prob)
    
    
    # Cost function: Add cost function to tensorflow graph
    
    cost = compute_cost(Z5, y_)
   
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        predict_op = tf.argmax(Z5, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            batch_xs, batch_ys = mnist.train.next_batch(minibatch_size)
            batch_xs = batch_xs.reshape((-1,28,28,1))
            seed = seed + 1

            _ , temp_cost = sess.run([optimizer, cost], feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.4})
                
            minibatch_cost = temp_cost 
                

            # Print the cost every 100 epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                train_accuracy = accuracy.eval({x:batch_xs, y_:batch_ys, keep_prob:1.0})
                print("Train Accuracy:", train_accuracy)
            if print_cost == True and epoch % 100 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        
        # Calculate accuracy on the test set
       
        test_images = mnist.test.images
        test_images = test_images.reshape((-1,28,28,1))
        test_accuracy = accuracy.eval({x: test_images, y_: mnist.test.labels, keep_prob:1.0})
        print("Test Accuracy:", test_accuracy)
                
        return test_accuracy, parameters


# In[ ]:


_, parameters = model(mnist,learning_rate=0.001, num_epochs=20000, minibatch_size=50)


# ## tensorflow中文社区中的示例代码

# In[ ]:


sess = tf.InteractiveSession()


# In[ ]:


# 为输入和输出建立placeholder
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# In[ ]:


# 权重初始化函数
def weight_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[ ]:


# 卷积和池化函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:


w_conv1 = weight_init([5, 5, 1, 32])
b_conv1 = bias_init([32])


# In[ ]:


x_image = tf.reshape(x, [-1, 28, 28, 1])


# In[ ]:


h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[ ]:


w_conv2 = weight_init([5, 5, 32, 64])
b_conv2 = bias_init([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[ ]:


w_fc1 = weight_init([7 * 7 * 64, 1024])
b_fc1 = bias_init([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


# In[ ]:


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_init([1024, 10])
b_fc2 = bias_init([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)


# In[ ]:


## 训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("setp %d, training accuracy %g"%(i, train_accuracy))
        
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    

test_data1, test_data2 = test_data[0:5000, :], test_data[5000:, :]
test_label1, test_label2 = mnist.test.labels[0:5000, :], mnist.test.labels[5000:, :]

test_accuracy1 = accuracy.eval(feed_dict={x:test_data1, y_:test_label1, keep_prob: 1.0})
test_accuracy2 = accuracy.eval(feed_dict={x:test_data2, y_:test_label2, keep_prob: 1.0})
test_accuracy = (test_accuracy1 + test_accuracy2) / 2
print ("test_accuracy %f"%test_accuracy)

