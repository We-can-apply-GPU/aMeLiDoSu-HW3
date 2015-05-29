import theano
import theano.tensor as T
from .layers import get_output

def mse(x, t):
    return (x - t) ** 2

class Objective(object):

    def __init__(self, input_layer, loss_function=mse, mode='mean'):
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target = T.matrix("target")
        self.mode = mode

    def get_loss(self, input=None, target=None, mode='mean', training=False, **kwargs):
        output = get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target

        losses = self.loss_function(output, target)

        real_loss = loss = losses.mean();
        if training:
            loss =  T.switch(T.lt(loss, theano.shared(100)), loss, 0.1*loss)
            loss =  T.switch(T.lt(loss, theano.shared(200)), loss, 0)
        xx = output.flatten()
        tt = target.flatten()
        accu = T.dot(xx, tt)/(T.sqrt((xx**2).sum())*T.sqrt((tt**2).sum()))

        return loss, real_loss - accu*100 + 1000, accu
