import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

        Args:
            - ``Z`` -- numpy array of any shape

        Returns:
            - ``A`` -- output of ``sigmoid(z)``, same shape as ``Z``
            - ``cache`` -- returns ``Z`` as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Implement the ``RELU`` function.

        Arguments:
            - ``Z`` -- Output of the linear layer, of any shape

        Returns:
            - ``A`` -- Post-activation parameter, of the same shape as Z
            - ``cache`` -- a python dictionary containing "A" ; stored for computing the backward pass efficiently

    .. math::
        ReLU(Z) = max(0, Z)
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache
