from collections import deque

import theano
import numpy as np

from .. import utils

__all__ = [
    "get_all_layers",
    "get_all_layers_old",
    "get_output",
    "get_output_shape",
    "get_all_params",
    "count_params",
    "get_all_param_values",
    "set_all_param_values",
    "get_all_bias_params",
    "get_all_non_bias_params",
]


def get_all_layers(layer, treat_as_input=None):
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


def get_all_layers_old(layer):
    if isinstance(layer, (list, tuple)):
        layers = list(layer)
    else:
        layers = [layer]
    layers_to_expand = list(layers)
    while len(layers_to_expand) > 0:
        current_layer = layers_to_expand.pop(0)
        children = []

        if hasattr(current_layer, 'input_layers'):
            children = current_layer.input_layers
        elif hasattr(current_layer, 'input_layer'):
            children = [current_layer.input_layer]

        # filter the layers that have already been visited, and remove None
        # elements (for layers without incoming layers)
        children = [child for child in children
                    if child not in layers and
                    child is not None]
        layers_to_expand.extend(children)
        layers.extend(children)

    return layers


def get_output(layer_or_layers, inputs=None, **kwargs):
    from .input import InputLayer
    from .base import MergeLayer
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
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = utils.as_theano_expression(inputs)
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
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

    from .input import InputLayer
    from .base import MergeLayer
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
        if len(all_shapes) > 1:
            raise ValueError("get_output_shape() was called with a single "
                             "input shape on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input shapes instead.")
        for input_layer in all_shapes:
            all_shapes[input_layer] = input_shapes
    # update layer-to-shape mapping by propagating the input shapes
    for layer in all_layers:
        if layer not in all_shapes:
            try:
                if isinstance(layer, MergeLayer):
                    input_shapes = [all_shapes[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    input_shapes = all_shapes[layer.input_layer]
            except KeyError:
                raise ValueError("get_output() was called without giving an "
                                 "input shape for the free-floating layer %r. "
                                 "Please call it with a dictionary mapping "
                                 "this layer to an input shape."
                                 % layer)
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


def get_all_bias_params(layer):
    return get_all_params(layer, regularizable=False)


def get_all_non_bias_params(layer):
    return get_all_params(layer, regularizable=True)


def count_params(layer, **tags):
    params = get_all_params(layer, **tags)
    shapes = [p.get_value().shape for p in params]
    counts = [np.prod(shape) for shape in shapes]
    return sum(counts)


def get_all_param_values(layer, **tags):
    params = get_all_params(layer, **tags)
    return [p.get_value() for p in params]


def set_all_param_values(layer, values, **tags):
    params = get_all_params(layer, **tags)
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))
    for p, v in zip(params, values):
        p.set_value(v)
