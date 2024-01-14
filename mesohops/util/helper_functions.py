import numpy as np
import scipy as sp

__title__ = "Helper Functions"
__author__ = "J. K. Lynd"
__version__ = "1.3"

def array_to_tuple(array):
    """
    Converts an inputted array to a tuple.

    Parameters
    ----------
    1. array : np.array
               Numpy array.

    Returns
    -------
    1. tuple : tuple
               Array in tuple form.
    """
    if sp.sparse.issparse(array):
        if array.getnnz() > 0:
            return tuple([tuple(l) for l in np.nonzero(array)])
        else:
            return tuple([])
    else:
        if len(array) > 0:
            return tuple([tuple(l) for l in np.nonzero(array)])
        else:
            return tuple([])