from matplotlib import pyplot 
import numpy as np 
import random

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_prime(z):
    return np.exp(-z) / np.square(1. + np.exp(-z))

class Network(object):
    
    # size = [1,2,3]   
    # first layer is input layer, and last is output layer
    # each number signifies numbre of neurons in the layer
    
    def __init__(self, size):
        
        self.num_layers = len(size)
        self.size = size

        self.biases = [np.random.randn(y,1) for y in size[1:]]
        # returns a list of np arrays of size (y,1) each initiliases with random numbers from SND
        # input layer has of course no bias

        self.weights = [np.random.randn(x, y) for x,y in zip(size[1:], size[:-1])]
        # self.weights = [np.random.randn(x + 1, x) for x in size[:-1]] -> incorrect
        # vertically, ie-weight leading to each neuron in smallest list
        ''' eg  [array([[-0.05662823],
                [ 1.64900572]]), array([[ 0.84107151, -0.38905865],
                [ 0.16385993, -0.22332297],
                [ 1.62589098,  0.93344726]])] '''


    def feedforward(self, a):
        # returns output of network when a is input array
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # mini-batch stochastic gradient descent
        # training_data is a list of tuples(x,y) where x is input, ie-image(ndarray of 784 pixels)
        # y is desired output, ie- digit
        # eta is learning rate
        ''' test_data, if provided then network will be evaluated against the test data after each epoch, and partial progress printed out. '''
        if test_data: 
            n_test = len(test_data) #no of tests
        n = len(training_data)
        for ii in range(epochs):

            # creating mini_batches after shuffling
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size]
                            for k in range(0,n, mini_batch_size)]

            for  mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epoch {0} : {1} / {2}".format(ii, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(ii))

    def update_mini_batch(self, mini_batch, eta):
        '''Updates network's weight and biases via backpropagation using this mini_batch'''
        '''Mini_batch is from training_data hence is also a list of tuples 
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) 
            # gradient according to this training example
            # ie- gradient of C_x
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # summing the gradients for all training examples of mini_batch to get gradient C ie- sum of C_x

        self.weights = [w-(eta/len(mini_batch))*nw
                                for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        # updating the weights and biases for this mini_batch total

    def backprop(self,x,y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
       
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        # feedforward
        activation = x  #input as of now
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # delta_L
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
            

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        ''' Return the vector of partial derivatives partial C_x / partial a for the output activations. '''
        return (output_activations - y)


import loader
training_data, validation_data, test_data = loader.load_data_wrapper()

net = Network([784,30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)