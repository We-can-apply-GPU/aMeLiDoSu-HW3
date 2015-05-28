from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from . import utils

__all__ = [
    "rmsprop",
]

def get_or_compute_grads(loss_or_grads, params):
    if isinstance(loss_or_grads, list):
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    #ref :
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))
    return updates
