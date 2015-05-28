import numpy as np
import theano
import theano.tensor as T

from .. import nonlinearities, init, utils

from .base import Layer
from .input import InputLayer
from .dense import DenseLayer
import helper
from settings import *

class OutputLayer(Layer):

    def __init__(self, incoming, name,num_units,W=init.Uniform(),
                 b=init.Constant(0.),nonlinearity=nonlinearities.identity, **kwargs):
        """
        An output layer for recurrent specifically.

        Parameters
        ----------
        - incoming : a class Layer instance (from RecurrentLayer)
        - num_units : int
            The number of units of the layer
        - W : Theano shared variable, numpy array or callable
            An initializer for the weights of the layer. If a shared variable or a
            numpy array is provided the shape should  be (num_inputs, num_units).
            See :meth:`Layer.create_param` for more information.
        - nonlinearity : callable or None
            default is softMax (as its name XD)
        """

        super(OutputLayer, self).__init__(incoming,name,**kwargs)
        self.num_units = num_units
        self.num_grams = self.input_shape[1]
        self.num_features = self.input_shape[2]
        self.W = self.add_param(W, (self.num_features, self.num_units), name="W")
        ##########Issue###############
        self.nonlinearity = nonlinearity


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1],self.num_units) 
        # BATCHSIZE * (# of grams) * (NUM_UNITS @ this layer)   

    def get_output_for(self, input, *args, **kwargs):
        #What is tensotdot?? @@
        #-->ref : http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
        act = T.tensordot(input, self.W,axes=1)
        act = act.reshape((self.input_shape[0],self.input_shape[1]*self.num_units),ndim=2)
        result = self.nonlinearity(act)
        result = result.reshape((self.input_shape[0], self.input_shape[1], self.num_units),ndim=3)
        return result
#for deep RNN
#class OutputLayer(Layer):
    #def __init__(self,incoming,name,num_units,W=init.Uniform(),
            #b = init.Constant(0.),nonlinearity = nonlinearities.identity,**kwargs):
        #input_shape = incoming.get_output_shape()

        #self.num_units = num_units
        #self.num_grams = input_shape[1]
        #self.num_features = input_shape[2]
        #self.nonlinearity = nonlinearity
        ###create deep RNN  3_layer
        #gramFeatureProd = self.num_grams * self.num_features
        #self.reshape_in = ReshapeLayer(InputLayer(input_shape),shape=(input_shape[0],gramFeatureProd))

        #self.l1 = DenseLayer(InputLayer(self.reshape_in.get_output_shape_for(input_shape)),
                             #num_units = gramFeatureProd)
        #self.l2 = DenseLayer(InputLayer(self.l1.get_output_shape()),
                             #num_units = gramFeatureProd)
        #self.out = DenseLayer(InputLayer(self.l2.get_output_shape()),
                             #num_units)

        #super(OutputLayer, self).__init__(incoming,name,**kwargs)
        ###########Issue###############


    #def get_output_shape_for(self, input_shape):
        #return (input_shape[0], input_shape[1],self.num_units) 
        ## BATCHSIZE * (# of grams) * (NUM_UNITS @ this layer)  

    #def get_output_for(self, input, *args, **kwargs):
        ##What is tensotdot?? @@
        ##-->ref : http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
        #propogate = self.reshape_in.get_output(input)
        #propogate = self.l1.get_output(propogate)
        #propogate = self.l2.get_output(propogate)
        #result = self.nonlinearity(self.out.get_output(propogate)) 

        ##act = T.tensordot(input, self.W,axes=1)
        ##act = act.reshape((self.input_shape[0],self.input_shape[1]*self.num_units),ndim=2)
        ##result = self.nonlinearity(act)
        #result = result.reshape((self.input_shape[0], self.input_shape[1], self.num_units),ndim=3)
        #return result
