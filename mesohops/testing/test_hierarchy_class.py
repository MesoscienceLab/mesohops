import numpy as np
from mesohops.dynamics.hops_hierarchy import HopsHierarchy as HHier
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxVec


__title__ = "test of hops hierarchy"
__author__ = "D. I. G. Bennett, Leonel Varvelo"
__version__ = "0.1"
__date__ = "Jan. 14, 2020"


def test_hierarchy_initialize_true():
    """
    This test checks whether an adaptive calculation (True) creates a list of tuples
    n_hmodes long
    """
    # initializing hops_hierarchy class
    hierarchy_param = {"MAXHIER": 4}
    system_param = {"N_HMODES": 4}
    HH = HHier(hierarchy_param, system_param)
    HH.initialize(True)  # makes the calculation adaptive
    aux_list = HH.auxiliary_list
    known_tuple = [AuxVec([], 4)]
    assert known_tuple == aux_list


def test_hierarchy_initialize_false():
    """
    This test checks whether a non-adaptive calculation (False) creates a list of tuples
    applied to a triangular filter
    """
    # initializing hops_hierarchy class
    hierarchy_param = {"MAXHIER": 2, "STATIC_FILTERS": []}
    system_param = {"N_HMODES": 2}
    HH = HHier(hierarchy_param, system_param)
    HH.initialize(False)
    aux_list = HH.auxiliary_list
    # known result triangular filtered list
    known_triangular_list = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
        AuxVec([(0, 1), (1, 1)], 2),
        AuxVec([(1, 2)], 2),
    ]
    assert known_triangular_list == aux_list


def test_filter_aux_list_triangular():
    """
    This functions checks that the TRIANGULAR filter is being properly applied
    """
    # initializing hops_hierarchy class
    hierarchy_param = {
        "MAXHIER": 2,
        "STATIC_FILTERS": [("Triangular", [[False, True], [1, 1]])],
    }
    system_param = {"N_HMODES": 2}
    HH = HHier(hierarchy_param, system_param)
    aux_list = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
        AuxVec([(0, 1), (1, 1)], 2),
        AuxVec([(1, 2)], 2),
    ]
    aux_list = HH.filter_aux_list(aux_list)
    # known result filtered triangular list
    known_triangular_tuple = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
    ]
    assert aux_list == known_triangular_tuple
    assert HH.param["STATIC_FILTERS"] == [("Triangular", [[False, True], [1, 1]])]


def test_filer_aux_list_longedge():
    hierarchy_param = {"MAXHIER": 2}
    system_param = {"N_HMODES": 2}
    HH = HHier(hierarchy_param, system_param)
    aux_list = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
        AuxVec([(0, 1), (1, 1)], 2),
        AuxVec([(1, 2)], 2),
    ]
    known_aux_list = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
        AuxVec([(1, 2)], 2),
    ]
    aux_list = HH.apply_filter(aux_list, "LongEdge", [[False, True], [2, 1]])
    assert aux_list == known_aux_list
    assert HH.param["STATIC_FILTERS"] == [("LongEdge", [[False, True], [2, 1]])]


def test_aux_index_true():
    """
    This function test the case where _aux_index returns the absolute index of a
    specific auxiliary member
    """
    hierarchy_param = {"MAXHIER": 4}
    system_param = {"N_HMODES": 4}
    HH = HHier(hierarchy_param, system_param)
    abs_index = HH._aux_index(
        AuxVec([(0, 1), (2, 1)], 4), True
    )  # True = absolute index
    # known result based on alpha numeric ordering
    known_index = 7
    assert abs_index == known_index


def test_aux_index_false():
    """
    This function test the case where _aux_index returns the relative index of a
    specific auxiliary member. It is important to note because of auxilary_list
    setter our auxiliary list gets rearranged into alpha numerical order and the
    return index is that of the relative list in alpha numerical order
    """
    hierarchy_param = {"MAXHIER": 4}
    system_param = {"N_HMODES": 4}
    HH = HHier(hierarchy_param, system_param)
    HH.auxiliary_list = [
        AuxVec([], 4),
        AuxVec([(2, 1), (3, 1)], 4),
        AuxVec([(0, 1), (1, 1)], 4),
        AuxVec([(0, 1), (2, 1)], 4),
        AuxVec([(1, 1), (3, 1)], 4),
    ]
    relative_index = HH._aux_index(AuxVec([(0, 1), (2, 1)], 4), False)
    # known result based on alpha numerical ordering
    known_index = 2
    assert relative_index == known_index


def test_const_aux_edge():
    """
     This function test whether const_aux_edge is properly creating an auxiliary index
     tuple for an edge node at a particular depth along a given mode.
    """
    hierarchy_param = {"MAXHIER": 4}
    system_param = {"N_HMODES": 4}
    HH = HHier(hierarchy_param, system_param)
    HH.initialize(True)
    tmp = HH._const_aux_edge(2, 1, 4)
    known_index_tuple = AuxVec([(2, 1)], 4)
    assert tmp == known_index_tuple


# a function created to be used to test define_triangular_hierarchy
def map_to_auxvec(list_aux):
    list_aux_vec = []
    for aux_values in list_aux:
        aux_key = np.where(aux_values)[0]
        list_aux_vec.append(
            AuxVec([tuple([key, aux_values[key]]) for key in aux_key], 4)
        )
    return list_aux_vec


# array used to test define_triangular_hierarchy
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


def test_define_triangular_hierarchy_4modes_4maxhier():
    """
    This function test whether define_triangular_hierarchy is properly defining
    a triangular hierarchy and outputting it to a filtered list
    """
    hierarchy_param = {"MAXHIER": 10}
    system_param = {"N_HMODES": 10}
    HH = HHier(hierarchy_param, system_param)
    assert set(HH.define_triangular_hierarchy(4, 4)) == set(aux_list_4_4)
    assert set(HH.define_triangular_hierarchy(2, 4)) == set(aux_list_2_4)
