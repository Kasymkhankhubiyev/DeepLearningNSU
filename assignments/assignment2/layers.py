import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.trace(np.matmul(W.T, W))  # L2(W) = l * tr(W.T * W)
    grad = 2 * reg_strength * W  # dl2(W) / dW = 2 * l * W

    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        # избавляемся от чрезмерно больших значений
        preds = predictions.copy() - np.max(predictions)

        # вычислим заранее знаменатель
        denominator = np.sum(np.exp(preds))
        output = np.exp(preds) / denominator

        return output
    else:
        preds = predictions.copy() - np.max(predictions, axis=1).reshape(-1, 1)
        output = np.exp(preds) / np.sum(np.exp(preds), axis=1).reshape(-1, 1)

        return output

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if len(probs.shape) == 1:
        return -np.log(probs[target_index])
    else:
        return -np.sum(np.log(probs[np.arange(len(probs)), target_index]))


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    _predictions = softmax(preds)
    loss = cross_entropy_loss(_predictions, target_index)
    dprediction = _predictions.copy()

    if len(preds.shape) == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(len(dprediction)), target_index] -= 1

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.dx = 0

    def forward(self, X):
        self.dx = np.where(X > 0, 1, 0)
        return np.where(X > 0, X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * self.dx

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        pred = np.dot(X, self.W.value) + self.B.value

        return pred

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_result = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        dB = 2 * np.mean(d_out, axis=0)  # удвоенное, потому что ReLU c коэф 1/2

        self.W.grad += dW
        self.B.grad += dB

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
