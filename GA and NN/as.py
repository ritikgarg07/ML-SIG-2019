
#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


#%%
def activation(z, derivative=False):
    """
    Sigmoid activation function:
    It handles two modes: normal and derivative mode.
    Applies a pointwise operation on vectors

    Parameters:
    ---
    z: pre-activation vector at layer l
        shape (n[l], batch_size)
    Returns:
    pontwize activation on each element of the input z
    """
    if derivative:
        return -np.exp(z) / np.square(np.exp(z) + 1)
    else:
        return 1 / (1 - np.exp(z))


#%%
def cost_function(y_true, y_pred):
    """
    Computes the Mean Square Error between a ground truth vector and a prediction vector
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost: a scalar value representing the loss
    """
    n = y_pred.shape[1]
    cost = (1. / (2 * n)) * np.sum((y_true - y_pred)**2)
    return cost


#%%
def cost_function_prime(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the activation of the output layer
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost_prime: derivative of the loss w.r.t. the activation of the output
    shape: (n[L], batch_size)
    """
    cost_prime = y_pred - y_true 
    # Calculate the derivative of the cost function
    return cost_prime


#%%
class NeuralNetwork(object):
    '''
    This is a custom neural netwok package built from scratch with numpy.
    The Neural Network as well as its parameters and training method and procedure will
    reside in this class.
    Parameters
    ---
    size: list of number of neurons per layer
    Examples
    ---
    >>> import NeuralNetwork
    >>> nn = NeuralNetwork([2, 3, 4, 1])

    This means :
    1 input layer with 2 neurons
    1 hidden layer with 3 neurons
    1 hidden layer with 4 neurons
    1 output layer with 1 neuron

    '''

    def __init__(self, size, seed=42):
        '''
        Instantiate the weights and biases of the network
        weights and biases are attributes of the NeuralNetwork class
        They are updated during the training
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        # biases are initialized randomly
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]

        # initialize the weights randomly
        """
        Be careful with the dimensions of the weights
        The dimensions of the weight of any particular layer will depend on the
        size of the current layer and the previous layer
        Example: Size = [16,8,4,2]
        The weight file will be a list with 3 matrices with shapes:
        (8,16) for weights connecting layers 1 (16) and 2(8)
        (4,8) for weights connecting layers 2 (8) and 4(4)
        (2,4) for weights connecting layers 3 (4) and 4(2)
        Each matrix will be initialized with random values
        """
        self.weights = np.array([np.random.rand(size[a], size[a-1]) for a in range(1, size.size) ])

    def forward(self, input):
        '''
        Perform a feed forward computation
        Parameters
        ---
        input: data to be fed to the network with
        shape: (input_shape, batch_size)
        Returns
        ---
        a: ouptut activation (output_shape, batch_size)
        pre_activations: list of pre-activations per layer
        each of shape (n[l], batch_size), where n[l] is the number
        of neuron at layer l
        activations: list of activations per layer
        each of shape (n[l], batch_size), where n[l] is the number
        of neuron at layer l
        '''
        a = input
        pre_activations = []
        activations = [a]
        # what does the zip function do?
        # It packs up weights and biases. For eg. if weights=[1, 2, 3] and biases=[4,5,6], Then zip(weights, biases) = [[1,4], [2, 5], [3,6]]         
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations

    """
    Resources:
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    https://hmkcode.github.io/ai/backpropagation-step-by-step/
    """

    def compute_deltas(self, pre_activations, y_true, y_pred):
        """
        Computes a list containing the values of delta for each layer using
        a recursion
        Parameters:
        ---
        pre_activations: list of of pre-activations. each corresponding to a layer
        y_true: ground truth values of the labels
        y_pred: prediction values of the labels
        Returns:
        ---
        deltas: a list of deltas per layer

        """

        # initialize array to store the derivatives
        delta = [0] * (len(self.size) - 1)

        # Calculate the delta for each layer
        # This is the first step in calculating the derivative
        #The last layer is calculated as derivative of cost function *  derivative of sigmoid ( pre-activations of last layer )
        delta[-1] = cost_function_prime(y_true, y_pred) * activation(pre_activations[-1], derivative=True)

        # Recursively calculate delta for each layer from the previous layer
        for l in range(len(deltas) - 2, -1, -1):
            # deltas of layer l depend on the weights of layer l and l+1 and on the sigmoid derivative of the pre-activations of layer l
            # Note that we use a dot product when multipying the weights and the deltas
            # Check their shapes to ensure that their shapes conform to the requiremnts (You may need to transpose some of the matrices)
            # The final shape of deltas of layer l must be the same as that of the activations of layer l
            # Check if this is true
            
            delta[l] = np.sum(np.dot(self.weights[l].T, delta[l+1])) * activation(pre_activations[l], True)
            # I added np.sum here as the shape of deltas and activations has to be same.
            # Why np.sum ? No sure....But according to me any method for reduction of dimension would work (accuracy might vary) 
        return deltas

    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network
        Parameters:
        ---
        deltas: list of deltas computed by compute_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of activations per layer
        Returns:
        ---
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network

        """
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            # Compute the derivatives of the weights and the biases from the delta values calculated earlier
            # dW_temp depends on the activations of layer l-1 and the deltas of layer l
            # dB_temp depends only on the deltas of layer l
            # Again be careful of the dimensions and ensure that the dW matrix has the same shape as W
            dW_temp = np.dot(activations[l-1], deltas[l] ) 
            db_temp = deltas[l]
            dW.append(dW_temp)
            db.append(np.expand_dims(db_temp.mean(axis=1), 1))
        return dW, db

    def plot_loss(self, epochs, train, test):
        """
        Plots the loss function of the train test data measured every epoch
        Parameters:
        ---
        epochs: number of epochs for training
        train: list of losses on the train set measured every epoch
        test: list of losses on the test set measured every epoch
        """

        plt.subplot(211)
        plt.title('Training Cost (loss)')
        plt.plot(range(epochs), train)

        plt.subplot(212)
        plt.title('Test Cost (loss)')
        plt.plot(range(epochs), test)

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def train(self,
              X,
              y,
              batch_size,
              epochs,
              learning_rate,
              validation_split=0.2,
              print_every=10):
        """
        Trains the network using the gradients computed by back-propagation
        Splits the data in train and validation splits
        Processes the training data by batches and trains the network using batch gradient descent
        Parameters:
        ---
        X: input data
        y: input labels
        batch_size: number of data points to process in each batch
        epochs: number of epochs for the training
        learning_rate: value of the learning rate
        validation_split: percentage of the data for validation
        print_every: the number of epochs by which the network logs the loss and accuracy metrics for train and validations splits
        plot_every: the number of epochs by which the network plots the decision boundary

        Returns:
        ---
        history: dictionary of train and validation metrics per epoch
            train_acc: train accuracy
            test_acc: validation accuracy
            train_loss: train loss
            test_loss: validation loss
        This history is used to plot the performance of the model
        """
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []

        # Read about the train_test_split function
        x_train, x_test, y_train, y_test = train_test_split(
            X.T,
            y.T,
            test_size=validation_split,
        )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

        epoch_iterator = range(epochs)

        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size) - 1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            batches_x = [
                x_train[:, batch_size * i:batch_size * (i + 1)]
                for i in range(0, n_batches)
            ]
            batches_y = [
                y_train[:, batch_size * i:batch_size * (i + 1)]
                for i in range(0, n_batches)
            ]

            train_losses = []
            train_accuracies = []

            test_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases]

            for batch_x, batch_y in zip(batches_x, batches_y):
                batch_y_pred, pre_activations, activations = self.forward(
                    batch_x)
                deltas = self.compute_deltas(pre_activations, batch_y,
                                             batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations,
                                            activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)

                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = accuracy_score(batch_y.T,
                                                batch_y_train_pred.T)
                train_accuracies.append(train_accuracy)

                batch_y_test_pred = self.predict(x_test)

                test_loss = cost_function(y_test, batch_y_test_pred)
                test_losses.append(test_loss)
                test_accuracy = accuracy_score(y_test.T, batch_y_test_pred.T)
                test_accuracies.append(test_accuracy)

            # weight update

            # What does the enumerate function do?
            # It add index to the object that is being enumerated and returns an enumerate object.
            
            for i, (dw_epoch,
                    db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                # Update the weights using the backpropagation algorithm implemented earlier
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))

            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))

            if e % print_every == 0:
                print(
                    'Epoch {} / {} | train loss: {} | train accuracy: {} | val loss : {} | val accuracy : {} '
                    .format(e, epochs, np.round(np.mean(train_losses), 3),
                            np.round(np.mean(train_accuracies), 3),
                            np.round(np.mean(test_losses), 3),
                            np.round(np.mean(test_accuracies), 3)))

        self.plot_loss(epochs, train_loss, test_loss)

        history = {
            'epochs': epochs,
            'train_loss': history_train_losses,
            'train_acc': history_train_accuracies,
            'test_loss': history_test_losses,
            'test_acc': history_test_accuracies
        }
        return history

    def predict(self, a):
        '''
        Use the current state of the network to make predictions
        Parameters:
        ---
        a: input data, shape: (input_shape, batch_size)
        Returns:
        ---
        predictions: vector of output predictions
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions


