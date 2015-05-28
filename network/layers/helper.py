from collections import deque

import theano
import numpy as np

from .. import utils
from .input import InputLayer
from .base import MergeLayer

def get_all_layers(layer, treat_as_input=None):
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    saw = set()
    did = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        saw.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in saw:
            # We haven't saw this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            saw.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in did:
                result.append(layer)
                did.add(layer)

    return result


def get_output(layer_or_layers, inputs=None, **kwargs):
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, utils.as_theano_expression(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        for input_layer in all_outputs:
            all_outputs[input_layer] = utils.as_theano_expression(inputs)
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            if isinstance(layer, MergeLayer):
                layer_inputs = [all_outputs[input_layer]
                                for input_layer in layer.input_layers]
            else:
                layer_inputs = all_outputs[layer.input_layer]
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def get_output_shape(layer_or_layers, input_shapes=None):
    # shortcut: return precomputed shapes if we do not need to propagate any
    if input_shapes is None or input_shapes == {}:
        try:
            return [layer.output_shape for layer in layer_or_layers]
        except TypeError:
            return layer_or_layers.output_shape

    # obtain topological ordering of all layers the output layer(s) depend on
    if isinstance(input_shapes, dict):
        treat_as_input = input_shapes.keys()
    else:
        treat_as_input = []

    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-shape mapping from all input layers
    all_shapes = dict((layer, layer.shape)
                      for layer in all_layers
                      if isinstance(layer, InputLayer) and
                      layer not in treat_as_input)
    # update layer-to-shape mapping from given input(s), if any
    if isinstance(input_shapes, dict):
        all_shapes.update(input_shapes)
    elif input_shapes is not None:
        for input_layer in all_shapes:
            all_shapes[input_layer] = input_shapes
    # update layer-to-shape mapping by propagating the input shapes
    for layer in all_layers:
        if layer not in all_shapes:
            if isinstance(layer, MergeLayer):
                input_shapes = [all_shapes[input_layer]
                                for input_layer in layer.input_layers]
            else:
                input_shapes = all_shapes[layer.input_layer]
            all_shapes[layer] = layer.get_output_shape_for(input_shapes)
    # return the output shape(s) of the requested layer(s) only
    try:
        return [all_shapes[layer] for layer in layer_or_layers]
    except TypeError:
        return all_shapes[layer_or_layers]


def get_all_params(layer, **tags):
    layers = get_all_layers(layer)
    params = sum([l.get_params(**tags) for l in layers], [])
    return utils.unique(params)


def get_all_param_values(layer, **tags):
    params = get_all_params(layer, **tags)
    return [p.get_value() for p in params]

def set_all_param_values(layer, values, **tags):
    params = get_all_params(layer, **tags)
    for p, v in zip(params, values):
        p.set_value(v)
