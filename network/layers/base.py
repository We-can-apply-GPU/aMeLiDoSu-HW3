from collections import OrderedDict
from .. import utils
__all__ = [
    "Layer",
    "MergeLayer",
]

# Layer base class

class Layer(object):
    def __init__(self, incoming, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = OrderedDict()

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    def get_params(self, **tags):
        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        return result

    def get_output_shape(self):
        return self.output_shape

    def get_output(self, input=None, **kwargs):
        from .helper import get_output
        return get_output(self, input, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def add_param(self, spec, shape, name=None, **tags):
        # prefix the param name with the layer name if it exists
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)

        param = utils.create_param(spec, shape, name)
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param

    def get_bias_params(self):
        return self.get_params(regularizable=False)

class MergeLayer(Layer):
    def __init__(self, incomings, name=None):
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.output_shape for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming for incoming in incomings]
        self.name = name
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.get_output_shape_for(self.input_shapes)
