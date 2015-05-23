import theano
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy, categorical_crossentropy
from .layers import get_output


def mse(x, t):
    return (x - t) ** 2


class Objective(object):
    _valid_aggregation = {None, 'mean', 'sum'}

    def __init__(self, input_layer, loss_function=mse, aggregation='mean'):
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, aggregation=None, **kwargs):
        network_output = get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target_var
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        if aggregation is None:
            aggregation = self.aggregation

        losses = self.loss_function(network_output, target)

        if aggregation is None or aggregation == 'mean':
            return losses.mean()
        elif aggregation == 'sum':
            return losses.sum()
        else:
            raise RuntimeError('This should have been caught earlier')


class MaskedObjective(object):
    _valid_aggregation = {None, 'sum', 'mean', 'normalized_sum'}

    def __init__(self, input_layer, loss_function=mse, aggregation='mean'):
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.mask_var = T.matrix("mask")
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             '\'normalized_sum\' or None,'
                             ' not {0}'.format(aggregation))
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, mask=None,
                 aggregation=None, **kwargs):
        network_output = get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target_var
        if mask is None:
            mask = self.mask_var

        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             '\'normalized_sum\' or None, '
                             'not {0}'.format(aggregation))

        # Get aggregation value passed to constructor if None
        if aggregation is None:
            aggregation = self.aggregation

        masked_losses = self.loss_function(network_output, target) * mask

        if aggregation is None or aggregation == 'mean':
            return masked_losses.mean()
        elif aggregation == 'sum':
            return masked_losses.sum()
        elif aggregation == 'normalized_sum':
            return masked_losses.sum() / mask.sum()
        else:
            raise RuntimeError('This should have been caught earlier')
