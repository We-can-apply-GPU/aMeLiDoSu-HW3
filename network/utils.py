import numpy as np

import theano
import theano.tensor as T

def floatX(arr):
    return np.asarray(arr, dtype=theano.config.floatX)

def shared_empty(dim=2, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))

def as_theano_expression(input):
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        return theano.tensor.constant(input)

def unique(l):
    new_list = []
    for el in l:
        if el not in new_list:
            new_list.append(el)

    return new_list

def create_param(spec, shape, name=None):
    if isinstance(spec, theano.compile.SharedVariable):
        return spec

    elif isinstance(spec, np.ndarray):
        return theano.shared(spec, name=name)

    elif hasattr(spec, '__call__'):
        arr = spec(shape)
        try:
            arr = floatX(arr)
        except Exception:
            raise RuntimeError("cannot initialize parameters!")
        if arr.shape != shape:
            raise RuntimeError("cannot initialize parameters!")
        return theano.shared(arr, name=name)

    else:
        raise RuntimeError("cannot initialize parameters!")
