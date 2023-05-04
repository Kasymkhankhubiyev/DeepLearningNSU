import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output),
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for _, param in self.params().items():
            param.grad = np.zeros_like(param.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # forward pass
        out = X.copy()
        for layer in self.layers:
            out = layer.forward(out)

        # backward pass
        loss, dout = softmax_with_cross_entropy(out, y)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for _, param in self.params().items():
            reg_loss, dgrad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += dgrad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = np.zeros(X.shape[0], int)

        output = X.copy()
        for layer in self.layers:
            output = layer.forward(output)

        return pred + np.argmax(output, axis=1)

    def params(self):
        result = {}

        for idx, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result[name] = param

        return result
