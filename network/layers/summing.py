import theano.tensor as T

from .base import MergeLayer

class SummingLayer(MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(SummingLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        output = T.zeros_like(inputs[0])
        output = T.inc_subtensor(output[1:], inputs[0][:-1])
        output = T.inc_subtensor(output[:-1], inputs[1][1:])
        output /= 2
        output = T.inc_subtensor(output[-1], inputs[0][-2]/2)
        output = T.inc_subtensor(output[0], inputs[1][1]/2)
        return output
