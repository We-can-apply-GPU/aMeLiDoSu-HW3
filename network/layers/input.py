from collections import OrderedDict

import theano
import theano.tensor as T

from .. import utils

from .base import Layer


__all__ = [
    "InputLayer",
]


class InputLayer(Layer):
    def __init__(self, shape, input_var=None, name=None, **kwargs):
        self.shape = shape
        ndim = len(shape)
        if input_var is None:
            # create the right TensorType for the given number of dimensions
            input_var_type = T.TensorType(theano.config.floatX, [False] * ndim)
            var_name = ("%s.input" % name) if name is not None else "input"
            input_var = input_var_type(var_name)
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        self.name = name
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape
