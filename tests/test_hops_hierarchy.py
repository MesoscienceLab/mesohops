import pytest
import numpy as np
from mesohops.basis.hops_aux import AuxiliaryVector as AuxVec
from mesohops.basis.hops_hierarchy import HopsHierarchy as HHier
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.exceptions import UnsupportedRequest


__title__ = 'Unit Tests for HopsHierarchy'
__author__ = 'D. I. G. Bennett, L. Varvelo, J. K. Lynd'
__version__ = '1.6'


def test_maxhier_overflow_warning():
    # This case tests that constructing a hierarchy with MAXHIER > 255
    # emits a warning about integer overflow.
    with pytest.warns(UserWarning, match='integer overflow'):
        HHier({'MAXHIER': 256}, {'N_HMODES': 2})


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
        "STATIC_FILTERS": [("Triangular", [[False, True], 1])],
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
        AuxVec([(0, 1), (1, 1)], 2),
    ]
    assert aux_list == known_triangular_tuple
    assert HH.param["STATIC_FILTERS"] == [("Triangular", [[False, True], 1])]


def test_filter_aux_list_longedge():
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
    aux_list = HH.apply_filter(aux_list, "LongEdge", [[False, True], 1])
    assert aux_list == known_aux_list
    assert HH.param["STATIC_FILTERS"] == [("LongEdge", [[False, True], 1])]


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



# a helper used to test the hierarchy builder functions
def map_to_auxvec(list_aux, n_hmodes):
    """
    Helper function that maps a list of auxiliary indexing vectors to a list of
    AuxVec objects with those indexing vectors.

    PARAMETERS
    ----------
    1. list_aux : list(list(int))
                  List of auxiliary indexing vectors in list form
    2. n_hmodes : int
                  Number of modes in the hierarchy

    RETURNS
    -------
    1. list_aux_vec : list(list(AuxVec))
                      List of AuxVec objects corresponding to the indexing vectors in
                      list_aux in a hierarchy with n_hmodes modes
    """

    list_aux_vec = []
    for aux_values in list_aux:
        aux_key = np.where(aux_values)[0]
        list_aux_vec.append(
            AuxVec([tuple([key, aux_values[key]]) for key in aux_key], n_hmodes)
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
    ], 4
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
    ], 2
)

def test_static_filter_init():
    """
    Tests that the static hierarchy filters throw an error if defined for the wrong
    number of modes.
    """
    # Define a HOPS trajectory in a non-special case
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
                   # Units: fs
                   "TAU": 1.0,  # Units: fs
                   }
    nsite = 10
    (g_0, w_0) = [10,150]
    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(loperator[i])
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])
    hs = np.zeros([nsite, nsite])
    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
                 "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
                 "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
                        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
                        'EARLY_INTEGRATOR_STEPS': 5,
                        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    hierarchy_param_mark = {'MAXHIER': 6,
                            'STATIC_FILTERS': [
                                ['Markovian', [True] * (len(gw_sysbath) - 1)],
                            ]}

    hierarchy_param_tri = {'MAXHIER': 6,
                           'STATIC_FILTERS': [
                               ['Triangular', [[True] * (len(gw_sysbath) - 1), 2]],
                           ]}

    hierarchy_param_le = {'MAXHIER': 6,
                          'STATIC_FILTERS': [
                              ['LongEdge', [[True] * (len(gw_sysbath) - 1), 2]],
                          ]}

    for filter_dict in [hierarchy_param_mark, hierarchy_param_tri, hierarchy_param_le]:
        with pytest.raises(UnsupportedRequest, match="The number of entries in the list "
                                                     "of static filter booleans does not"):
            hops = HOPS(sys_param,
                           noise_param=noise_param,
                           hierarchy_param=filter_dict,
                           eom_param=eom_param,
                           integration_param=integrator_param, )
            hops.make_adaptive(0, 0)
            hops.initialize(psi_0)

        with pytest.raises(UnsupportedRequest, match="The number of entries in the list "
                                                     "of static filter booleans does not"):
            hops_ad = HOPS(sys_param,
                           noise_param=noise_param,
                           hierarchy_param=filter_dict,
                           eom_param=eom_param,
                           integration_param=integrator_param, )
            hops_ad.make_adaptive(1e-3, 1e-3)
            hops_ad.initialize(psi_0)
            hops_ad.propagate(10.0, 2.0)

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
    system_param = {"N_HMODES": 4}
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


# ============================================================
# TEST SUITE: define_rectangular_hierarchy()
# ============================================================

# ------------------------------------------------------------
# TEST: Correct auxiliaries for two hierarchy modes
# ------------------------------------------------------------
def test_define_rect_hier_two_modes():
    # This case tests n_hmodes=2, maxhier=2 producing all 9 Cartesian
    # product combinations. Uses sorted list comparison (not set) to
    # catch duplicates in the output.
    list_aux = HHier.define_rectangular_hierarchy(2, 2)
    known_list = map_to_auxvec([
        [0, 0], [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2],
    ], 2)
    assert sorted(list_aux) == sorted(known_list), (
        'Two-mode rectangular hierarchy does not match expected Cartesian product'
    )


# ------------------------------------------------------------
# TEST: Rectangular is a strict superset of triangular for
#       n_hmodes > 1
# ------------------------------------------------------------
@pytest.mark.parametrize('n_hmodes, maxhier', [(3, 2), (2, 3)])
def test_define_rect_hier_superset_of_triangular(n_hmodes, maxhier):
    # This case tests that the rectangular hierarchy is a strict superset
    # of the triangular hierarchy for n_hmodes > 1.
    list_rect = HHier.define_rectangular_hierarchy(n_hmodes, maxhier)
    list_tri = HHier.define_triangular_hierarchy(n_hmodes, maxhier)
    set_rect = set(list_rect)
    set_tri = set(list_tri)
    assert set_tri.issubset(set_rect), (
        'Triangular hierarchy should be a subset of rectangular hierarchy'
    )
    assert len(set_rect) > len(set_tri), (
        'Rectangular hierarchy should be strictly larger than triangular '
        f'for n_hmodes > 1 (rect={len(set_rect)}, tri={len(set_tri)})'
    )


# ------------------------------------------------------------
# TEST: Single-mode rectangular and triangular are identical
# ------------------------------------------------------------
def test_define_rect_hier_single_mode_matches_triangular():
    # This case tests that for n_hmodes=1, rectangular and triangular
    # truncation produce identical hierarchies.
    n_hmodes = 1
    maxhier = 4
    list_rect = HHier.define_rectangular_hierarchy(n_hmodes, maxhier)
    list_tri = HHier.define_triangular_hierarchy(n_hmodes, maxhier)
    assert set(list_rect) == set(list_tri), (
        'Rectangular and triangular hierarchies should be identical for '
        'n_hmodes=1'
    )


# ------------------------------------------------------------
# TEST: Zero vector is always present in rectangular hierarchy
# ------------------------------------------------------------
def test_define_rect_hier_contains_zero_vector():
    # This case tests that the zero auxiliary vector (vacuum state)
    # is always present in the rectangular hierarchy.
    n_hmodes = 3
    maxhier = 2
    list_aux = HHier.define_rectangular_hierarchy(n_hmodes, maxhier)
    zero_aux = AuxVec([], n_hmodes)
    assert zero_aux in list_aux, (
        'Zero auxiliary vector should always be present in rectangular hierarchy'
    )


# ============================================================
# TEST SUITE: HopsHierarchy.initialize() with rectangular truncation
# ============================================================

# ------------------------------------------------------------
# TEST: Non-adaptive initialization with rectangular truncation
#       produces the rectangular hierarchy
# ------------------------------------------------------------
def test_initialize_rect_trunc_nonadaptive():
    # This case tests that initializing with TRUNCATION_METHOD='rectangular'
    # produces the same hierarchy as define_rectangular_hierarchy.
    hierarchy_param = {'MAXHIER': 2, 'TRUNCATION_METHOD': 'rectangular'}
    system_param = {'N_HMODES': 2}
    HH = HHier(hierarchy_param, system_param)
    HH.initialize(False)
    expected = HHier.define_rectangular_hierarchy(2, 2)
    assert len(HH.auxiliary_list) == len(expected), (
        f'Expected {len(expected)} auxiliaries, got {len(HH.auxiliary_list)}'
    )
    assert set(HH.auxiliary_list) == set(expected), (
        'Initialized rect hierarchy does not match define_rectangular_hierarchy'
    )


# ------------------------------------------------------------
# TEST: Rectangular truncation with Markovian filter warns and
#       produces correct filtered hierarchy
# ------------------------------------------------------------
def test_rect_trunc_with_markovian_filter():
    # This case tests that rectangular truncation with a Markovian filter
    # warns about tensor-specific filters, applies the filter, and
    # produces the correct hierarchy.
    hierarchy_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'rectangular',
        'STATIC_FILTERS': [('Markovian', [False, True])],
    }
    system_param = {'N_HMODES': 2}
    HH = HHier(hierarchy_param, system_param)
    with pytest.warns(UserWarning, match='tensor-specific filters'):
        HH.initialize(False)
    # Markovian on mode 1 removes any aux with mode 1 active at total
    # depth > 1. From the 9-element rectangular hierarchy, only 4 survive.
    known = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
    ]
    assert sorted(HH.auxiliary_list) == sorted(known)


# ------------------------------------------------------------
# TEST: Rectangular + Markovian filter matches triangular +
#       Markovian filter (filtered modes produce same hierarchy)
# ------------------------------------------------------------
def test_rect_trunc_markovian_matches_triangular():
    # This case tests that applying the same Markovian filter to
    # rectangular and triangular hierarchies produces the same result,
    # since filtering collapses the extra rectangular elements.
    markov_filter = [('Markovian', [False, True])]
    system_param = {'N_HMODES': 2}
    rect_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'rectangular',
        'STATIC_FILTERS': list(markov_filter),
    }
    tri_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'triangular',
        'STATIC_FILTERS': list(markov_filter),
    }
    HH_rect = HHier(rect_param, system_param)
    HH_tri = HHier(tri_param, system_param)
    with pytest.warns(UserWarning, match='tensor-specific filters'):
        HH_rect.initialize(False)
    HH_tri.initialize(False)
    assert sorted(HH_rect.auxiliary_list) == sorted(HH_tri.auxiliary_list)


# ------------------------------------------------------------
# TEST: Rectangular truncation with Triangular filter produces
#       correct filtered hierarchy
# ------------------------------------------------------------
def test_rect_trunc_with_triangular_filter():
    # This case tests that applying a Triangular filter to a rectangular
    # hierarchy correctly prunes auxiliaries. The Triangular filter on
    # mode 1 with kmax=1 restricts the total depth in filtered modes
    # to be <= 1.
    hierarchy_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'rectangular',
        'STATIC_FILTERS': [('Triangular', [[False, True], 1])],
    }
    system_param = {'N_HMODES': 2}
    HH = HHier(hierarchy_param, system_param)
    with pytest.warns(UserWarning, match='tensor-specific filters'):
        HH.initialize(False)
    # Triangular filter on mode 1 with kmax_2=1: keeps only auxiliaries
    # where the sum of depth in filtered modes (mode 1) is <= 1.
    for aux in HH.auxiliary_list:
        depth_mode1 = aux.get(1, 0)
        assert depth_mode1 <= 1, (
            f'Triangular filter failed: aux {aux} has mode-1 depth '
            f'{depth_mode1}, expected <= 1'
        )
    # Should be smaller than unfiltered rectangular
    rect_unfiltered = HHier.define_rectangular_hierarchy(2, 2)
    assert len(HH.auxiliary_list) < len(rect_unfiltered)


# ------------------------------------------------------------
# TEST: Rectangular truncation with LongEdge filter produces
#       correct filtered hierarchy
# ------------------------------------------------------------
def test_rect_trunc_with_longedge_filter():
    # This case tests that applying a LongEdge filter to a rectangular
    # hierarchy correctly prunes auxiliaries. LongEdge with kdepth=1
    # on mode 1: beyond total depth 1, only edge terms (single-mode
    # depth) are kept for filtered modes.
    hierarchy_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'rectangular',
        'STATIC_FILTERS': [('LongEdge', [[False, True], 1])],
    }
    system_param = {'N_HMODES': 2}
    HH = HHier(hierarchy_param, system_param)
    with pytest.warns(UserWarning, match='tensor-specific filters'):
        HH.initialize(False)
    known = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
        AuxVec([(0, 2)], 2),
        AuxVec([(1, 2)], 2),
    ]
    assert sorted(HH.auxiliary_list) == sorted(known)


# ------------------------------------------------------------
# TEST: Rectangular + LongEdge matches triangular + LongEdge
# ------------------------------------------------------------
def test_rect_trunc_longedge_matches_triangular():
    # This case tests that applying the same LongEdge filter to
    # rectangular and triangular hierarchies produces the same result.
    longedge_filter = [('LongEdge', [[False, True], 1])]
    system_param = {'N_HMODES': 2}
    rect_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'rectangular',
        'STATIC_FILTERS': list(longedge_filter),
    }
    tri_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'triangular',
        'STATIC_FILTERS': list(longedge_filter),
    }
    HH_rect = HHier(rect_param, system_param)
    HH_tri = HHier(tri_param, system_param)
    with pytest.warns(UserWarning, match='tensor-specific filters'):
        HH_rect.initialize(False)
    HH_tri.initialize(False)
    assert sorted(HH_rect.auxiliary_list) == sorted(HH_tri.auxiliary_list)


# ------------------------------------------------------------
# TEST: Rectangular truncation with multiple filters
#       (Markovian + Triangular)
# ------------------------------------------------------------
def test_rect_trunc_with_combined_filters():
    # This case tests that applying multiple filters to a rectangular
    # hierarchy correctly chains the filtering. Markovian on mode 1
    # followed by Triangular on mode 0 with kmax=1.
    hierarchy_param = {
        'MAXHIER': 2,
        'TRUNCATION_METHOD': 'rectangular',
        'STATIC_FILTERS': [
            ('Markovian', [False, True]),
            ('Triangular', [[True, False], 1]),
        ],
    }
    system_param = {'N_HMODES': 2}
    HH = HHier(hierarchy_param, system_param)
    with pytest.warns(UserWarning, match='tensor-specific filters'):
        HH.initialize(False)
    # Markovian on mode 1 gives: {}, {0:1}, {1:1}, {0:2}
    # Triangular on mode 0 with kmax=1 keeps only depth <= 1 in mode 0
    # That removes {0:2}, leaving: {}, {0:1}, {1:1}
    known = [
        AuxVec([], 2),
        AuxVec([(0, 1)], 2),
        AuxVec([(1, 1)], 2),
    ]
    assert sorted(HH.auxiliary_list) == sorted(known)
