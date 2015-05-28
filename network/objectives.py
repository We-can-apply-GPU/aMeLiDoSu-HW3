import theano
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy, categorical_crossentropy
from .layers import get_output


def mse(x, t):
    return (x - t) ** 2

class Objective(object):

    def __init__(self, input_layer, loss_function=mse, mode='mean'):
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target = T.matrix("target")
        self.mode = mode

    def get_loss(self, input=None, target=None, mode='mean', **kwargs):
        output = get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target

        losses = self.loss_function(output, target)

        if mode == 'mean':
            return losses.mean()
        else: 
            return losses.sum()
