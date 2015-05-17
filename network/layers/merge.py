import theano.tensor as T

from .base import MergeLayer


__all__ = [
    "ElemwiseSumLayer",
]

class ElemwiseSumLayer(MergeLayer):
    """
    This layer performs an elementwise sum of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.

    Notes
    -----------
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """

    def __init__(self, incomings, coeffs=1, **kwargs):
        super(ElemwiseSumLayer, self).__init__(incomings, **kwargs)
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)
        self.coeffs = coeffs

    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        output = None
        for coeff, input in zip(self.coeffs, inputs):
            if coeff != 1:
                input *= coeff
            if output is not None:
                output += input
            else:
                output = input
        return output