import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities, init, utils

from lasagne.layers.base import Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper

class RecurrentSoftmaxLayer(Layer):
    def __init__(self, incoming, num_units,W=init.Uniform()
                ,b=init.Constant(0.),nonlinearity=nonlinearities.softmax, **kwargs):
        
        #####Basic Method
        ###shape maybe weird@@
        super(RecurrentSoftmaxLayer, self).__init__(incoming,**kwargs)
	self.num_units = num_units
        self.num_inputs = self.input_shape[1]   #???
	#self.num_time_steps = self.input_shape[1]
	#self.num_features = self.input_shape[2]	
	self.W = self.create_param(W, (self.num_inputs, self.num_units), name="W")

    def get_output_shape_for(self, input_shape):
	return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
	activation = T.dot(input, self.W )
	return self.nonlinearity(activation)
	

class CustomRecurrentLayer(Layer):
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, trace_steps=-1):
        '''
            - input_to_hidden : Layer connecting input to the hidden state
            - hidden_to_hidden : Layer connecting previous hidden state to new state
            - backwards : boolean
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
            - trace_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''
        super(CustomRecurrentLayer, self).__init__(incoming)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.trace_steps = trace_steps
        self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        (num_batches, self.num_units) = self.input_to_hidden.get_output_shape()

        # Initialize hidden state
        self.hid_init = self.create_param(hid_init, (num_batches, self.num_units))

    def get_params(self):

        #return a list
        params = (helper.get_all_params(self.input_to_hidden) +
                helper.get_all_params(self.hidden_to_hidden))

        if self.learn_init:
            return params + self.get_init_params()  # return the initial pars
        else:
            return params

    def get_init_params(self):
        return [self.hid_init]

    def get_bias_params(self):
        return (helper.get_all_bias_params(self.input_to_hidden) +
                helper.get_all_bias_params(self.hidden_to_hidden))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input,*args, **kwargs):
        #see mask~~


        # Create single recurrent computation step function
        def step(layer_input, prev_hid):
            return self.nonlinearity(
                self.input_to_hidden.get_output(layer_input) +
                self.hidden_to_hidden.get_output(prev_hid))

        return output


class RecurrentLayer(CustomRecurrentLayer):
    def __init__(self, incoming, num_units, W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, trace_steps=-1):
        '''
        Create a recurrent layer.
            - W_in_to_hid
            - W_hid_to_hid
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : If True, process the sequence backwards
            - learn_init : If True, initial hidden values can be learned
            - trace_steps :
                Number of steps to trace in BPTT
                If -1 -> through whole sequence
        '''
        input_shape = incoming.get_output_shape()
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the first dimension
        in_to_hid = DenseLayer(InputLayer((input_shape[0],) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=nonlinearity)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((input_shape[0], num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=nonlinearity)

        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=backwards,
            trace_steps=trace_steps)
