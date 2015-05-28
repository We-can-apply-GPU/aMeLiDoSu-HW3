import numpy as np
import theano
import theano.tensor as T

from .. import nonlinearities, init, utils
from .base import Layer

class RecurrentLayer(Layer):
    def __init__(self, incoming, name, num_units,
                 W_i2h=init.GlorotUniform(),
                 W_h2h=init.GlorotUniform(),
                 h_init=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 backwards=False, trace_steps=-1):
        '''
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
        super(RecurrentLayer, self).__init__(incoming, name)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.W_i2h = self.add_param(W_i2h, (num_inputs, num_units), name="W_i2h")
        self.W_h2h = self.add_param(W_h2h, (num_units, num_units), name="W_h2h")
        self.h_init = self.add_param(h_init, (1, num_units))
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.backwards = backwards
        self.trace_steps = trace_steps

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        # Single recurrent computation
        #ref:http://deeplearning.net/software/theano/library/scan.html
        def step(layer_input, prev_hidden_state):
            return self.nonlinearity(
                T.dot(layer_input, self.W_i2h) + T.dot(prev_hidden_state, self.W_h2h))

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

        #Refer to the order od theano.scan ~ seqs -> output_info -> nonseqs
        output, _ = theano.scan(fn=step, sequences=input,
                                go_backwards=self.backwards,
                                outputs_info=[self.h_init],
                                truncate_gradient=self.trace_steps)

        if self.backwards:
            output = output[::-1]  # reverse the gram to noraml index~~

        return output
