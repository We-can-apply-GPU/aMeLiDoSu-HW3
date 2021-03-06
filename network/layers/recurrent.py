import numpy as np
import theano
import theano.tensor as T

from .. import nonlinearities, init, utils

from .base import Layer
from .input import InputLayer
from .dense import DenseLayer
import helper
from settings import *

class CustomRecurrentLayer(Layer):

    def __init__(self, incoming,name ,input_to_hid, hid_to_hid,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,trace_steps=-1):
        '''
        An bottom  layer for RecurrentLayer.

        Parameters
        ----------
        From slide p.6
            - input_to_hid
            - hid_to_hid
            - backwards : boolean
                If True, process the sequence backwards(used in biDirectional)
            - trace_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''
        super(CustomRecurrentLayer, self).__init__(incoming,name)

        self.input_to_hid = input_to_hid
        self.hid_to_hid = hid_to_hid
        self.nonlinearity = nonlinearity
        self.backwards = backwards
        self.trace_steps = trace_steps

        # Get batchSize and num_units at high level
        # num_batches == input_shape[0]
        # Initialize hidden state
        (n_batch,self.num_units) = self.input_to_hid.get_output_shape()
        self.hid_init = self.add_param(hid_init,self.input_to_hid.get_output_shape())

    def get_params(self):
        params = (helper.get_all_params(self.input_to_hid) +
                  helper.get_all_params(self.hid_to_hid)   + [self.hid_init] )
        return params   #return a list

    def get_all_non_bias_params(self):
        return (helper.get_all_non_bias_params(self.input_to_hid) +
                helper.get_all_non_bias_params(self.hid_to_hid)   + [self.hid_init] )

    def get_bias_params(self):
        return (helper.get_all_bias_params(self.input_to_hid) +
                helper.get_all_bias_params(self.hid_to_hid))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[1],self.num_units)

    def get_output_for(self, input,*args, **kwargs):

        # Single recurrent computation
        def step(layer_input, prev_hidden_state):
            return self.nonlinearity(
                self.input_to_hid.get_output(layer_input) +
                self.hid_to_hid.get_output(prev_hidden_state))
        #ref:http://deeplearning.net/software/theano/library/scan.html

        #No non-changing variable -> thus,no non_sequence
        #outputs_info is used for initialization
        #truncate_gradient is the number of steps to use in truncated BPTT. 
        #If you compute gradients through a scan op, 
        #they are computed using backpropagation through time. 
        #By providing a different value then -1, you choose to use truncated BPTT 
        #instead of classical BPTT, 
        #where you go for only truncate_gradient number of steps back in time.

        # Input should be provided as (n_batch, nGrams, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (nGrams, n_batch, n_features)
        #print(shape(input))
        input = input.dimshuffle(1, 0, 2)

        #Refer to the order od theano.scan ~ seqs -> output_info -> nonseqs
        output = theano.scan(fn=step, sequences=input,
                             go_backwards=self.backwards,
                             outputs_info=[self.hid_init],
                             truncate_gradient=self.trace_steps)[0]
        # Now, dimshuffle back to (n_batch, nGrams, n_features))
        output = output.dimshuffle(1, 0, 2)

        if self.backwards:
            output = output[:, ::-1, :]  # reverse the gram to noraml index~~

        return output

class RecurrentLayer(CustomRecurrentLayer):
    def __init__(self, incoming,name ,num_units, W_i=init.Uniform(),
                 W_h=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,trace_steps=-1):
        '''
        An top layer for RecurrentLayer.

        Parameters
        ----------
        Create a recurrent layer.
            - W_i
            - W_h
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : If True, process the sequence backwards
            - trace_steps :
                Number of steps to trace in BPTT
                If -1 -> through whole sequence
        '''

        input_shape = incoming.get_output_shape()

        #One gram in each step
        input_to_hid = DenseLayer(InputLayer((input_shape[0],input_shape[2])),
                                  num_units,W = W_i,b=b,nonlinearity = nonlinearity)

        hid_to_hid = DenseLayer(InputLayer((input_shape[0], num_units)),
                                num_units,W = W_h,nonlinearity=nonlinearity)

        super(RecurrentLayer, self).__init__(
            incoming, name,input_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards,trace_steps=trace_steps)
