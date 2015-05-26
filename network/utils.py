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
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))

def unique(l):
    new_list = []
    for el in l:
        if el not in new_list:
            new_list.append(el)

    return new_list

def create_param(spec, shape, name=None):
    if isinstance(spec, theano.compile.SharedVariable):
        if spec.ndim != len(shape):
            raise RuntimeError("shared variable has %d dimensions, "
                               "should be %d" % (spec.ndim, len(shape)))
        return spec

    elif isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise RuntimeError("parameter array has shape %s, should be "
                               "%s" % (spec.shape, shape))
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
