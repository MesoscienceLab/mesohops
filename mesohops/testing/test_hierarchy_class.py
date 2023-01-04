import numpy as np
from mesohops.dynamics.hops_hierarchy import HopsHierarchy as HHier
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxVec


__title__ = "test of hops hierarchy"
__author__ = "D. I. G. Bennett, L. Varvelo, J. K. Lynd"
__version__ = "1.2"
__date__ = "Jan. 14, 2020"


def test_hierarchy_initialize_true():
    """
    Tests whether an adaptive calculation (True) creates a list of tuples n_hmodes long.
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
    Tests whether a non-adaptive calculation (False) creates a list of tuples
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

def test_filter_aux_list_markovian():
    """
    Tests that the Markovian filter is being properly applied.
    """

    hierarchy_param = {
        "MAXHIER": 2,
        "STATIC_FILTERS": [("Markovian", [False, True])],
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
    # known result filtered Markovian list
    known_markovian_tuple = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
    ]
    assert aux_list == known_markovian_tuple
    assert HH.param["STATIC_FILTERS"] == [("Markovian", [False, True])]



def test_filter_aux_list_triangular():
    """
    Tests that the TRIANGULAR filter is being properly applied.
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
    """
    Tests that the LONGEDGE filter is being properly applied.
    """

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


def test_aux_index_relative():
    """
    Tests the case where _aux_index returns the relative index of a
    specific auxiliary member. It is important to note because of auxilary_list
    setter our auxiliary list gets rearranged into alpha numerical order and the
    return index is that of the relative list in alpha numerical order. Note that if
    the AuxVec object has an index of None, the index of the identical AuxVec in the
    HopsHierarchy.auxiliary_list will be returned. Otherwise, the AuxVec.index value
    will be returned.
    """

    hierarchy_param = {"MAXHIER": 4}
    system_param = {"N_HMODES": 4}

    # Test case: AuxVec.index is None
    HH = HHier(hierarchy_param, system_param)
    HH.auxiliary_list = [
        AuxVec([], 4),
        AuxVec([(2, 1), (3, 1)], 4),
        AuxVec([(0, 1), (1, 1)], 4),
        AuxVec([(0, 1), (2, 1)], 4),
        AuxVec([(1, 1), (3, 1)], 4),
    ]
    relative_index = HH._aux_index(AuxVec([(0, 1), (2, 1)], 4))
    # known result based on alpha numerical ordering
    known_index = 2
    assert relative_index == known_index

    # Test case: AuxVec.index is not None
    test_aux = AuxVec([(0, 1), (2, 1)], 4)
    HH = HHier(hierarchy_param, system_param)
    HH.auxiliary_list = [
        AuxVec([], 4),
        AuxVec([(2, 1), (3, 1)], 4),
        AuxVec([(0, 1), (1, 1)], 4),
        test_aux,
        AuxVec([(1, 1), (3, 1)], 4),
    ]
    relative_index = HH._aux_index(test_aux)
    assert relative_index == test_aux._index

    # Show that if the index is manually set, _aux_index returns the value manually set
    test_aux._index = 10
    assert HH._aux_index(test_aux) == 10


def test_const_aux_edge():
    """
    Tests whether const_aux_edge is properly creating an auxiliary index tuple for
    an edge node at a particular depth along a given mode.
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
    """
    Helper function that maps a list of auxiliary indexing vectors to a list of
    AuxVec objects with those indexing vectors.

    PARAMETERS
    ----------
    1. list_aux : list(list(int))
                  List of auxiliary indexing vectors in list form

    RETURNS
    -------
    1. list_aux_vec : list(list(AuxVec))
                      List of AuxVec objects corresponding to the indexing vectors in
                      list_aux in a hierarchy with 4 modes
    """

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
    Tests that define_triangular_hierarchy is properly defining a triangular
    hierarchy and outputting it to a filtered list.
    """

    hierarchy_param = {"MAXHIER": 10}
    system_param = {"N_HMODES": 10}
    HH = HHier(hierarchy_param, system_param)
    assert set(HH.define_triangular_hierarchy(4, 4)) == set(aux_list_4_4)
    assert set(HH.define_triangular_hierarchy(2, 4)) == set(aux_list_2_4)


def test_define_markovian_filtered_triangular_hierarchy():
    """
    Tests that the define_markovian_filtered_triangular_hierarchy function that is
    automatically called to lower memory burdens when the first static filter is
    Markovian outputs the correct Markovian-filtered hierarchy. Test cases: number of
    modes equal to hierarchy depth, number of modes greater than the hierarchy depth,
    number of modes less than the hierarchy depth.
    """

    hierarchy_param = {"MAXHIER": 10,
                       "STATIC_FILTERS": [['Markovian', [False, True, True, True]]]}
    system_param = {"N_HMODES": 10}
    HH = HHier(hierarchy_param, system_param)
    mark_list_aux = HH.define_markovian_filtered_triangular_hierarchy(4, 4, [False,
                                                                    True, True, True])
    nonmark_list_aux = HH.define_triangular_hierarchy(4, 4)
    filtered_list_aux = HH.filter_aux_list(nonmark_list_aux)
    assert set(filtered_list_aux) == set(mark_list_aux)
    assert set(nonmark_list_aux) != set(mark_list_aux)

    # Case: number of modes greater than depth
    mark_list_aux = HH.define_markovian_filtered_triangular_hierarchy(4, 3, [False,
                                                                    True, True, True])
    nonmark_list_aux = HH.define_triangular_hierarchy(4, 3)
    filtered_list_aux = HH.filter_aux_list(nonmark_list_aux)
    assert set(filtered_list_aux) == set(mark_list_aux)
    assert set(nonmark_list_aux) != set(mark_list_aux)

    # Case: number of modes less than depth
    mark_list_aux = HH.define_markovian_filtered_triangular_hierarchy(4, 5, [False,
                                                                    True, True, True])
    nonmark_list_aux = HH.define_triangular_hierarchy(4, 5)
    filtered_list_aux = HH.filter_aux_list(nonmark_list_aux)
    assert set(filtered_list_aux) == set(mark_list_aux)
    assert set(nonmark_list_aux) != set(mark_list_aux)

def test_define_markovian_and_LE_filtered_triangular_hierarchy():
    """
    Tests that the define_markovian_and_LE_filtered_triangular_hierarchy function
    that is automatically called to lower memory burdens when the first static filter is
    Markovian and the second is LongEdge outputs the correct
    Markovian-and-LongEdge-filtered hierarchy. Test cases: number of
    modes equal to hierarchy depth, number of modes greater than the hierarchy depth,
    number of modes less than the hierarchy depth.

    NOTE: This is technically an integrated test! This function is never run without
    being wrapped in HopsHierarchy.filter_aux_list, because it does not actually
    filter out the longedge auxiliaries with total depth beyond the depth limit set
    out by the parameter LE_depth.
    """

    hierarchy_param = {"MAXHIER": 10,
                       "STATIC_FILTERS": [['Markovian', [False, True, False, True]],
                                           ["LongEdge", [[False, False, True, True],
                                                         [3, 1]
                                                         ]]]}
    system_param = {"N_HMODES": 10}
    HH = HHier(hierarchy_param, system_param)
    prefiltered_list_aux = HH.filter_aux_list(
        HH.define_markovian_and_LE_filtered_triangular_hierarchy(4,
             4, [False, True, True, True], [False, False, True, True], 1))
    unfiltered_list_aux = HH.define_triangular_hierarchy(4, 4)
    postfiltered_list_aux = HH.filter_aux_list(unfiltered_list_aux)
    assert set(prefiltered_list_aux) == set(postfiltered_list_aux)
    assert set(prefiltered_list_aux) != set(unfiltered_list_aux)

    # Case: number of modes greater than depth
    prefiltered_list_aux = HH.filter_aux_list(
        HH.define_markovian_and_LE_filtered_triangular_hierarchy(4,
             3, [False, True, True, True], [False, False, True, True], 1))
    unfiltered_list_aux = HH.define_triangular_hierarchy(4, 3)
    postfiltered_list_aux = HH.filter_aux_list(unfiltered_list_aux)
    assert set(prefiltered_list_aux) == set(postfiltered_list_aux)
    assert set(prefiltered_list_aux) != set(unfiltered_list_aux)

    # Case: number of modes less than depth
    prefiltered_list_aux = HH.filter_aux_list(
        HH.define_markovian_and_LE_filtered_triangular_hierarchy(4,
             5, [False, True, True, True], [False, False, True, True], 1))
    unfiltered_list_aux = HH.define_triangular_hierarchy(4, 5)
    postfiltered_list_aux = HH.filter_aux_list(unfiltered_list_aux)
    assert set(prefiltered_list_aux) == set(postfiltered_list_aux)
    assert set(prefiltered_list_aux) != set(unfiltered_list_aux)


def test_add_connections():
    """
    Tests that the add_connections function is properly adding connections to all
    vectors 1 off at only 1 index.
    """

    # [0, 0, 0]
    vector_000 = AuxVec([], 3)
    # [1, 0, 0]
    vector_100 = AuxVec([(0, 1)], 3)
    # [0, 1, 0]
    vector_010 = AuxVec([(1, 1)], 3)
    # [0, 0, 1]
    vector_001 = AuxVec([(2, 1)], 3)
    # [2, 0, 0]
    vector_200 = AuxVec([(0, 2)], 3)
    # [0, 2, 0]
    vector_020 = AuxVec([(1, 2)], 3)
    # [0, 0, 2]
    vector_002 = AuxVec([(2, 2)], 3)
    # [1, 1, 0]
    vector_110 = AuxVec([(0, 1), (1, 1)], 3)
    # [1, 0, 1]
    vector_101 = AuxVec([(0, 1), (2, 1)], 3)
    # [0, 1, 1]
    vector_011 = AuxVec([(1, 1), (2, 1)], 3)
    # [3, 0, 0]
    vector_300 = AuxVec([(0,3)], 3)
    # [0, 3, 0]
    vector_030 = AuxVec([(1,3)], 3)
    # [0, 0, 3]
    vector_003 = AuxVec([(2,3)], 3)

    list_all_vectors = [vector_000, vector_100, vector_010, vector_001, vector_200,
                        vector_020, vector_002, vector_110, vector_101, vector_011,
                        vector_300, vector_030, vector_003]


    hier_param = {"MAXHIER": 3}
    sys_param = {"N_HMODES": 3}
    test_hier = HHier(hier_param, sys_param)

    test_hier.auxiliary_list = list_all_vectors

    # Test that all vectors form connections only to those that differ by 1 exactly
    # at only 1 index, with the dictionary key at that index.
    assert vector_000._dict_aux_m1 == {}
    assert vector_000._dict_aux_p1 == {0:vector_100, 1:vector_010, 2:vector_001}
    assert vector_100._dict_aux_m1 == {0:vector_000}
    assert vector_100._dict_aux_p1 == {0:vector_200, 1:vector_110, 2:vector_101}
    assert vector_010._dict_aux_m1 == {1:vector_000}
    assert vector_010._dict_aux_p1 == {0:vector_110, 1:vector_020, 2:vector_011}
    assert vector_001._dict_aux_m1 == {2:vector_000}
    assert vector_001._dict_aux_p1 == {0:vector_101, 1:vector_011, 2:vector_002}
    assert vector_200._dict_aux_m1 == {0:vector_100}
    assert vector_200._dict_aux_p1 == {0:vector_300}
    assert vector_020._dict_aux_m1 == {1:vector_010}
    assert vector_020._dict_aux_p1 == {1:vector_030}
    assert vector_002._dict_aux_m1 == {2:vector_001}
    assert vector_002._dict_aux_p1 == {2:vector_003}
    assert vector_110._dict_aux_m1 == {0:vector_010, 1:vector_100}
    assert vector_110._dict_aux_p1 == {}
    assert vector_101._dict_aux_m1 == {0:vector_001, 2:vector_100}
    assert vector_101._dict_aux_p1 == {}
    assert vector_011._dict_aux_m1 == {1:vector_001, 2:vector_010}
    assert vector_011._dict_aux_p1 == {}
    assert vector_300._dict_aux_m1 == {0:vector_200}
    assert vector_300._dict_aux_p1 == {}
    assert vector_030._dict_aux_m1 == {1:vector_020}
    assert vector_030._dict_aux_p1 == {}
    assert vector_003._dict_aux_m1 == {2:vector_002}
    assert vector_003._dict_aux_p1 == {}
