"""
Non-linear activation functions for artificial neurons.
"""

import theano.tensor.nnet

# sigmoid
def sigmoid(x):
    return theano.tensor.nnet.sigmoid(x)

# softmax (row-wise)
def softmax(x):
    return theano.tensor.nnet.softmax(x)

# tanh
def tanh(x):
    return theano.tensor.tanh(x)

# rectify
def rectify(x):
    return 0.5 * (x + abs(x))

# linear
def linear(x):
    return x

identity = linear
