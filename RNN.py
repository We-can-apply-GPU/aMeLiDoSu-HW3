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

    ###Todo : 
    #How to use batch for training???(add another dimension?)

    def __init__(self, incoming, num_units,W=init.Uniform()
                ,b=init.Constant(0.),nonlinearity=nonlinearities.softmax, **kwargs):
     
        """
        An output layer for recurrent specifically.

        Parameters
        ----------
        - incoming : a class Layer instance 
        - num_units : int
            The number of units of the layer
        - W : Theano shared variable, numpy array or callable
            An initializer for the weights of the layer. If a shared variable or a
            numpy array is provided the shape should  be (num_inputs, num_units).
            See :meth:`Layer.create_param` for more information.
        - nonlinearity : callable or None
            default is softMax (as its name XD)
        """

        super(RecurrentSoftmaxLayer, self).__init__(incoming,**kwargs)
	self.num_units = num_units
        self.num_features = self.input_shape[1]
	self.W = self.create_param(W, (self.num_features, self.num_units), name="W")
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
	return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
	result = T.dot(input, self.W )
	return self.nonlinearity(result)
	

class CustomRecurrentLayer(Layer):
    def __init__(self, incoming, hidden_layer, prev_hidden_layer,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, trace_steps=-1):
        
        '''
        An bottom  layer for RecurrentLayer.

        Parameters
        ----------
        From slide p.6 
            - hidden_layer : a Layer connecting
            - prev_hidden_layer : Layer connecting previous hidden state to new state
            - backwards : boolean
                If True, process the sequence backwards(used in biDirectional)
            - learn_init : boolean
                If True, initial hidden values are learned
            - trace_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''
        super(CustomRecurrentLayer, self).__init__(incoming)

        self.hidden_layer = hidden_layer
        self.prev_hidden_layer = prev_hidden_layer
        self.nonlinearity = nonlinearity
        self.backwards = backwards
        self.learn_init = learn_init
        self.trace_steps = trace_steps

        # Get batchSize and num_units at high level
        #num_batches == input_shape[0]
        self.output_shape = self.hidden_layer.get_output_shape()

        # Initialize hidden state
        self.hid_init = self.create_param(hid_init, self.output_shape)

    def get_params(self):

        #return a list
        params = (helper.get_all_params(self.hidden_layer) +
                  helper.get_all_params(self.prev_hidden_layer))

        if self.learn_init:
            return params + self.get_init_params()  # return the initial params
        else:
            return params

    def get_init_params(self):
        return [self.hid_init]

    def get_bias_params(self):
        return (helper.get_all_bias_params(self.hidden_layer) +
                helper.get_all_bias_params(self.prev_hidden_layer))

    def get_output_shape_for(self, input_shape):
        return (self.output_shape)

    def get_output_for(self, input,*args, **kwargs):
        #see mask~~


        # Single recurrent computation
        def step(layer_input, prev_hidden_state):
            return self.nonlinearity(
                self.hidden_layer.get_output(layer_input) +
                self.prev_hidden_layer.get_output(prev_hidden_state))
        #ref:http://deeplearning.net/software/theano/library/scan.html

        #No non-changing variable -> thus,no non_sequence
        #outputs_info is used for initialization
        #truncate_gradient – truncate_gradient is the number of steps to use in truncated BPTT. 
        #If you compute gradients through a scan op, 
        #they are computed using backpropagation through time. 
        #By providing a different value then -1, you choose to use truncated BPTT 
        #instead of classical BPTT, 
        #where you go for only truncate_gradient number of steps back in time.


        #if self.backwards:
            #sequences = [input, mask]
            #step_act = step_back
        #else:
        sequences = input
        step_act = step
        output = theano.scan(fn=step_act, sequences=sequences,
                             go_backwards=self.backwards,
                             outputs_info=self.hid_init,
                             truncate_gradient=self.trace_steps)
        return output

class RecurrentLayer(CustomRecurrentLayer):
    def __init__(self, incoming, num_units, W_i=init.Uniform(),
                 W_h=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, trace_steps=-1):
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
            - learn_init : If True, initial hidden values can be learned
            - trace_steps :
                Number of steps to trace in BPTT
                If -1 -> through whole sequence
        '''

        input_shape = incoming.get_output_shape()
        
        hidden_layer = DenseLayer(incoming,num_units,W = W_i,b=b,nonlinearity = nonlinearity)
        prev_hidden_layer = DenseLayer(hidden_layer,num_units,W = W_h,
                                       b=None,nonlinearity=nonlinearity)

        super(RecurrentLayer, self).__init__(
            incoming, hidden_layer, prev_hidden_layer, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=backwards,
            trace_steps=trace_steps)
