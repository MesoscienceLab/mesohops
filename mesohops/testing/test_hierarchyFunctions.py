import numpy as np
import mesohops.dynamics.hierarchy_functions as HF
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxVec

__title__ = "Test of hierarchyFunctions"
__author__ = "D. I. G. Bennett"
__version__ = "1.4"
__date__ = "Feb. 9, 2019"


def map_to_auxvec(list_aux):
    """
    This function takes a list of auxiliaries and outputs the associated
    auxiliary-objects.

    PARAMETERS
    ----------
    1. list_aux :  list
                   list of values corresponding to the auxiliaries in a basis.
    RETURNS
    -------
    1. list_aux_vec :  list
                       list of auxiliary-objects corresponding to these auxiliaries.
    """
    list_aux_vec = []
    for aux_values in list_aux:
        aux_key = np.where(aux_values)[0]
        list_aux_vec.append(
            AuxVec([tuple([key, aux_values[key]]) for key in aux_key], 4)
        )
    return list_aux_vec


aux_list_4_4 = map_to_auxvec(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 1, 1],
        [0, 0, 2, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 2, 0, 0],
        [1, 1, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 3],
        [0, 0, 1, 2],
        [0, 0, 2, 1],
        [0, 0, 3, 0],
        [0, 1, 0, 2],
        [0, 1, 1, 1],
        [0, 1, 2, 0],
        [1, 0, 0, 2],
        [1, 0, 1, 1],
        [1, 0, 2, 0],
        [0, 2, 0, 1],
        [0, 2, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [2, 0, 0, 1],
        [2, 0, 1, 0],
        [0, 3, 0, 0],
        [1, 2, 0, 0],
        [2, 1, 0, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 4],
        [0, 0, 1, 3],
        [0, 0, 2, 2],
        [0, 0, 3, 1],
        [0, 0, 4, 0],
        [0, 1, 0, 3],
        [0, 1, 1, 2],
        [0, 1, 2, 1],
        [0, 1, 3, 0],
        [1, 0, 0, 3],
        [1, 0, 1, 2],
        [1, 0, 2, 1],
        [1, 0, 3, 0],
        [0, 2, 0, 2],
        [0, 2, 1, 1],
        [0, 2, 2, 0],
        [1, 1, 0, 2],
        [1, 1, 1, 1],
        [1, 1, 2, 0],
        [2, 0, 0, 2],
        [2, 0, 1, 1],
        [2, 0, 2, 0],
        [0, 3, 0, 1],
        [0, 3, 1, 0],
        [1, 2, 0, 1],
        [1, 2, 1, 0],
        [2, 1, 0, 1],
        [2, 1, 1, 0],
        [3, 0, 0, 1],
        [3, 0, 1, 0],
        [0, 4, 0, 0],
        [1, 3, 0, 0],
        [2, 2, 0, 0],
        [3, 1, 0, 0],
        [4, 0, 0, 0],
    ]
)

aux_list_2_4 = map_to_auxvec(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 0],
        [1, 1],
        [0, 2],
        [3, 0],
        [2, 1],
        [1, 2],
        [0, 3],
        [4, 0],
        [3, 1],
        [2, 2],
        [1, 3],
        [0, 4],
    ]
)

aux_list_4_4_tf = map_to_auxvec(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 1, 1],
        [0, 0, 2, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 2, 0, 0],
        [1, 1, 0, 0],
        [2, 0, 0, 0],
        # [0, 0, 0, 3],
        [0, 0, 1, 2],
        [0, 0, 2, 1],
        [0, 0, 3, 0],
        # [0, 1, 0, 2],
        [0, 1, 1, 1],
        [0, 1, 2, 0],
        [1, 0, 0, 2],
        [1, 0, 1, 1],
        [1, 0, 2, 0],
        # [0, 2, 0, 1],
        [0, 2, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [2, 0, 0, 1],
        [2, 0, 1, 0],
        # [0, 3, 0, 0],
        [1, 2, 0, 0],
        [2, 1, 0, 0],
        [3, 0, 0, 0],
        # [0, 0, 0, 4],
        # [0, 0, 1, 3],
        [0, 0, 2, 2],
        [0, 0, 3, 1],
        [0, 0, 4, 0],
        # [0, 1, 0, 3],
        # [0, 1, 1, 2],
        [0, 1, 2, 1],
        [0, 1, 3, 0],
        # [1, 0, 0, 3],
        [1, 0, 1, 2],
        [1, 0, 2, 1],
        [1, 0, 3, 0],
        # [0, 2, 0, 2],
        # [0, 2, 1, 1],
        [0, 2, 2, 0],
        # [1, 1, 0, 2],
        [1, 1, 1, 1],
        [1, 1, 2, 0],
        [2, 0, 0, 2],
        [2, 0, 1, 1],
        [2, 0, 2, 0],
        # [0, 3, 0, 1],
        # [0, 3, 1, 0],
        # [1, 2, 0, 1],
        [1, 2, 1, 0],
        [2, 1, 0, 1],
        [2, 1, 1, 0],
        [3, 0, 0, 1],
        [3, 0, 1, 0],
        # [0, 4, 0, 0],
        # [1, 3, 0, 0],
        [2, 2, 0, 0],
        [3, 1, 0, 0],
        [4, 0, 0, 0],
    ]
)

aux_list_4_4_lef = map_to_auxvec(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 1, 1],
        [0, 0, 2, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 2, 0, 0],
        [1, 1, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 3],
        # [0, 0, 1, 2],
        # [0, 0, 2, 1],
        [0, 0, 3, 0],
        # [0, 1, 0, 2],
        # [0, 1, 1, 1],
        # [0, 1, 2, 0],
        # [1, 0, 0, 2],
        # [1, 0, 1, 1],
        [1, 0, 2, 0],
        # [0, 2, 0, 1],
        # [0, 2, 1, 0],
        # [1, 1, 0, 1],
        # [1, 1, 1, 0],
        # [2, 0, 0, 1],
        [2, 0, 1, 0],
        [0, 3, 0, 0],
        # [1, 2, 0, 0],
        # [2, 1, 0, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 4],
        # [0, 0, 1, 3],
        # [0, 0, 2, 2],
        # [0, 0, 3, 1],
        [0, 0, 4, 0],
        # [0, 1, 0, 3],
        # [0, 1, 1, 2],
        # [0, 1, 2, 1],
        # [0, 1, 3, 0],
        # [1, 0, 0, 3],
        # [1, 0, 1, 2],
        # [1, 0, 2, 1],
        [1, 0, 3, 0],
        # [0, 2, 0, 2],
        # [0, 2, 1, 1],
        # [0, 2, 2, 0],
        # [1, 1, 0, 2],
        # [1, 1, 1, 1],
        # [1, 1, 2, 0],
        # [2, 0, 0, 2],
        # [2, 0, 1, 1],
        [2, 0, 2, 0],
        # [0, 3, 0, 1],
        # [0, 3, 1, 0],
        # [1, 2, 0, 1],
        # [1, 2, 1, 0],
        # [2, 1, 0, 1],
        # [2, 1, 1, 0],
        # [3, 0, 0, 1],
        [3, 0, 1, 0],
        [0, 4, 0, 0],
        # [1, 3, 0, 0],
        # [2, 2, 0, 0],
        # [3, 1, 0, 0],
        [4, 0, 0, 0],
    ]
)

aux_list_4_4_df = map_to_auxvec(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 1, 1],
        [0, 0, 2, 0],
        [0, 2, 0, 0],
        [1, 1, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 3],
        [0, 0, 3, 0],
        [0, 3, 0, 0],
        [3, 0, 0, 0],
    ]
)


# Test base functions
# -------------------
def test_filter_aux_triangular():
    """
    test the triangular filter
    """
    assert set(HF.filter_aux_triangular(aux_list_4_4, [False, True] * 2, 2)) == set(
        aux_list_4_4_tf
    )


def test_filter_aux_longedge():
    """
    test the longedge filter
    """
    assert set(HF.filter_aux_longedge(aux_list_4_4, [False, True] * 2, 2)) == set(
        aux_list_4_4_lef
    )


def test_check_markovian():
    """
    checks to make sure an auxiliary is non markovian or an allowed markovian aux
    """
    aux_0000 = AuxVec([], 4)
    aux_1000 = AuxVec([(0, 1)], 4)
    aux_0100 = AuxVec([(1, 1)], 4)
    aux_1010 = AuxVec([(0, 1), (2, 1)], 4)
    aux_2000 = AuxVec([(0, 2)], 4)
    aux_0101 = AuxVec([(1, 1), (3, 1)], 4)
    aux_1001 = AuxVec([(0, 1), (3, 1)], 4)
    aux_1111 = AuxVec([(0, 1), (1, 1), (2, 1), (3, 1)], 4)
    list_bool = [False, True, False, True]
    # aux_1000 check
    markov_bool = HF.check_markovian(aux_0000, list_bool)
    known_bool = True
    assert markov_bool == known_bool
    # aux_1000 check
    markov_bool = HF.check_markovian(aux_1000, list_bool)
    known_bool = True
    assert markov_bool == known_bool
    # aux_0100 check
    markov_bool = HF.check_markovian(aux_0100, list_bool)
    known_bool = True
    assert markov_bool == known_bool
    # aux_1010 check
    markov_bool = HF.check_markovian(aux_1010, list_bool)
    known_bool = True
    assert markov_bool == known_bool
    # aux_2000 check
    markov_bool = HF.check_markovian(aux_2000, list_bool)
    known_bool = True
    assert markov_bool == known_bool
    # aux_0101 check
    markov_bool = HF.check_markovian(aux_0101, list_bool)
    known_bool = False
    assert markov_bool == known_bool
    # aux_1001 check
    markov_bool = HF.check_markovian(aux_1001, list_bool)
    known_bool = False
    assert markov_bool == known_bool
    # aux_1111 check
    markov_bool = HF.check_markovian(aux_1111, list_bool)
    known_bool = False
    assert markov_bool == known_bool


def test_filter_markovian():
    """
    test to make sure filter_markovian is properly filtering auxiliaries
    """
    # Note: Does not like the zero vector
    list_bool = [False, True, False, True]
    list_aux = [
        AuxVec([], 4),
        AuxVec([(3, 1)], 4),
        AuxVec([(2, 1)], 4),
        AuxVec([(2, 1), (3, 1)], 4),
        AuxVec([(1, 1)], 4),
        AuxVec([(1, 1), (3, 1)], 4),
        AuxVec([(1, 1), (2, 1)], 4),
        AuxVec([(1, 1), (2, 1), (3, 1)], 4),
        AuxVec([(0, 1)], 4),
        AuxVec([(0, 1), (3, 1)], 4),
        AuxVec([(0, 1), (2, 1)], 4),
        AuxVec([(0, 1), (2, 1), (3, 1)], 4),
        AuxVec([(0, 1), (1, 1)], 4),
        AuxVec([(0, 1), (1, 1), (3, 1)], 4),
        AuxVec([(0, 1), (1, 1), (2, 1)], 4),
        AuxVec([(0, 1), (1, 1), (2, 1), (3, 1)], 4),
    ]
    list_markov = HF.filter_markovian(list_aux, list_bool)
    known_list = [
        AuxVec([], 4),
        AuxVec([(3, 1)], 4),
        AuxVec([(2, 1)], 4),
        AuxVec([(1, 1)], 4),
        AuxVec([(0, 1)], 4),
        AuxVec([(0, 1), (2, 1)], 4),
    ]
    assert list_markov == known_list
