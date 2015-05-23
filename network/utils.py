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


def one_hot(x, m=None):
    if m is None:
        m = T.cast(T.max(x) + 1, 'int32')

    return T.eye(m)[T.cast(x, 'int32')]


def unique(l):
    new_list = []
    for el in l:
        if el not in new_list:
            new_list.append(el)

    return new_list


def as_tuple(x, N):
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if len(X) != N:
        raise ValueError("input must be a single value "
                         "or an iterable with length {0}".format(N))

    return X


def compute_norms(array, norm_axes=None):
    ndim = array.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(array.ndim)
        )

    norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms


def create_param(spec, shape, name=None):
    if isinstance(spec, theano.compile.SharedVariable):
        # We cannot check the shape here, the shared variable might not be
        # initialized correctly yet. We can check the dimensionality
        # though. Note that we cannot assign a name here. We could assign
        # to the `name` attribute of the shared variable, but we shouldn't
        # because the user may have already named the variable and we don't
        # want to override this.
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
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return an "
                               "array-like value")
        if arr.shape != shape:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return a value "
                               "with the correct shape")
        return theano.shared(arr, name=name)

    else:
        raise RuntimeError("cannot initialize parameters: 'spec' is not "
                           "a numpy array, a Theano shared variable, or a "
                           "callable")
