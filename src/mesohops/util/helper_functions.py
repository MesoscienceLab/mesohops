import numpy as np
import scipy as sp

__title__ = "Helper Functions"
__author__ = "J. K. Lynd"
__version__ = "1.6"

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
               Array in tuple form ((r1, r2, r3,...), (c1, c2, c3...)) where (rn,
               cn) is the row, column position of the nth nonzero entry in the array.
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

def index_raveler(ls):
    """
    Ravels indices to be used for indexing sparse arrays.

    Parameters
    ----------
    1. ls  : list(int)
             List of integers [i, j, k...]

    Returns
    -------
    1. list_raveled : list(list(int))
                      Raveled list of integers [[i], [j], [k]...]
    """
    return [[i] for i in ls]