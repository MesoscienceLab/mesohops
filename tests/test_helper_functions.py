import numpy as np
import scipy as sp
from mesohops.util.helper_functions import array_to_tuple

__title__ = "Test of Helper Functions"
__author__ = "J. K. Lynd"
__version__ = "1.3"

def test_array_to_tuple():
    """ This function test whether an array properly turns into a tuple"""
    array = np.zeros([4,4])
    array[0,0] = 1.0
    tuple_from_sparse_array = array_to_tuple(sp.sparse.coo_matrix(array))
    tuple_from_dense_array = array_to_tuple(array)
    known_tuple = (tuple([0]), tuple([0]))
    assert tuple_from_sparse_array == known_tuple
    assert tuple_from_dense_array == known_tuple