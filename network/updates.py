from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from . import utils

def get_or_compute_grads(loss_or_grads, params):
    if isinstance(loss_or_grads, list):
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)

def rnn_update(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    tmp = theano.shared(np.float32(0))
    cnt = theano.shared(np.float32(0))
    for grad in grads:
        tmp += T.sqrt((grad**2).mean())
        cnt += 1
    tmp /= cnt
    prod = T.switch(T.lt(tmp, theano.shared(0.5)), theano.shared(np.float32(0.01)), theano.shared(np.float32(0.01))/tmp)
    for param, grad in zip(params, grads):
        grad *= prod
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))
    return updates
