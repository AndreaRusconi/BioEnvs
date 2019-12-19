
import tensorflow as tf
import numpy as np 
import foolbox

# Parameters
learning_rate = 0.000
batch_size = 256
display_step = 1

# Network Parameters
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons
n_input = 24
#n_classes = 10 # MNIST total classes (0-9 digits)
output = 6.0
# tf Graph input
X = tf.compat.v1.placeholder("float", [None, n_input])
Y = tf.compat.v1.placeholder("float", [None, output])
weights1 = np.load('modello/parameterss/model/pi/fc0/kernel:0.npy')
wheights2 = np.load('modello/parameterss/model/pi/fc1/kernel:0.npy')
output = np.load('modello/parameterss/model/pi/dense/kernel:0.npy')
bias1 = np.load('modello/parameterss/model/pi/fc0/bias:0.npy')
bias2 = np.load('modello/parameterss/model/pi/fc1/bias:0.npy')
output_bias = np.load('modello/parameterss/model/pi/dense/bias:0.npy')
# Store layers weight & bias

weights = {
    'h1': tf.Variable(weights1),
    'h2': tf.Variable(wheights2),
    'out': tf.Variable(output)
}
biases = {
    'b1': tf.Variable(bias1),
    'b2': tf.Variable(bias2),
    'out': tf.Variable(output_bias)
}

print(weights['h1'])
# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 64 neurons
    layer_1 = tf.add(tf.matmul(x, (weights['h1'])), biases['b1'])
    # Hidden fully connected layer with 64 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)


# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
  
    foolbox.v1.attacks.BoundaryAttack(model)




    """
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
	
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))"""