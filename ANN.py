'''
Feed Forward
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) .................. > weight > output layer

Backward Propagation 
Compare output to intended output > cost function (cross entropy)
Minimize cost using optimization techniques (SGD, AdamOptimizer,..)

Feed Forward + Backward Propagation = epoch

What is one-hot ?
In our current case, mnist has output of handwritten digits ranging from 0-9
0 => [1,0,0,0,0,0,0,0,0,0,0]
1 => [0,1,0,0,0,0,0,0,0,0,0]
2 => [0,0,1,0,0,0,0,0,0,0,0]
3 => [0,0,0,1,0,0,0,0,0,0,0]
4 => [0,0,0,0,1,0,0,0,0,0,0] and so on...........
'''

# MNIST Dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Loading the MNIST Dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Number of Hidden Layer Neurons in Layer 1
N_HL_1 = 500
# Number of Hidden Layer Neurons in Layer 2
N_HL_2 = 500
# Number of Hidden Layer Neurons in Layer 3
N_HL_3 = 500
# Number of Input Features
N_INPUT = 784
# Number of Output Classes
N_CLASSES = 10
# Batch Size For AdamOptimizer
BATCH_SIZE = 100

# .placeholder with size doesnt allow any other value of different dimension to enter
x = tf.placeholder('float', [None,784])
y = tf.placeholder('float')

def feed_forward(data):
    # Initializing random values for weights     
    HL_1 = {
        'weights':tf.Variable(tf.random_normal([N_INPUT,N_HL_1])),
        'biases' :tf.Variable(tf.random_normal([N_HL_1]))
    }
    HL_2 = {
        'weights': tf.Variable(tf.random_normal([N_HL_1, N_HL_2])),
        'biases': tf.Variable(tf.random_normal([N_HL_2]))
    }
    HL_3 = {
        'weights': tf.Variable(tf.random_normal([N_HL_2, N_HL_3])),
        'biases': tf.Variable(tf.random_normal([N_HL_3]))
    }
    O_LAYER = {
        'weights': tf.Variable(tf.random_normal([N_HL_3, N_CLASSES])),
        'biases': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    # weight*input_data + biases
    l1 = tf.add(tf.matmul(data, HL_1['weights']),HL_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, HL_2['weights']),HL_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, HL_3['weights']),HL_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, O_LAYER['weights']) + O_LAYER['biases']

    return output

def train_nn(x):
    prediction = feed_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10
    # Running the session    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))


train_nn(x)
