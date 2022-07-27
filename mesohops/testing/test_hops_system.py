import os
import numpy as np
import scipy as sp
from pyhops.dynamics.hops_system import HopsSystem as HSystem
from pyhops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp

__title__ = "test for System Class"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.2"
__date__ = "Jan. 15, 2020"

# HOPS SYSTEM PARAMETERS
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 25000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

loperator = np.zeros([4, 4, 4], dtype=np.float64)
gw_sysbath = []
lop_list = []
for i in range(nsite):
    loperator[i, i, i] = 1.0
    gw_sysbath.append([g_0, w_0])
    lop_list.append(sp.sparse.coo_matrix(loperator[i]))
    gw_sysbath.append([-1j * np.imag(g_0), 500.0])
    lop_list.append(loperator[i])

hs = np.zeros([nsite, nsite])
hs[0, 1] = 40
hs[1, 0] = 40
hs[1, 2] = 10
hs[2, 1] = 10
hs[2, 3] = 40
hs[3, 2] = 40

sys_param = {
    "HAMILTONIAN": np.array(hs, dtype=np.complex128),
    "GW_SYSBATH": gw_sysbath,
    "L_HIER": lop_list,
    "L_NOISE1": lop_list,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": gw_sysbath,
}
HS = HSystem(sys_param)


def test_initialize_system_dict():
    """
    test to hops system dictionary is properly initialized
    """
    param_nstates = HS.param["NSTATES"]
    known_nstates = 4
    assert param_nstates == known_nstates

    param_nhmodes = HS.param["N_HMODES"]
    known_nhmodes = 8
    assert param_nhmodes == known_nhmodes

    param_g = HS.param["G"]
    path = os.path.realpath(__file__)
    path = path[: -len("test_hops_system.py")] + "/known_param_g.npy"
    known_param_g = np.load(path)
    assert np.allclose(param_g, known_param_g)

    param_w = HS.param["W"]
    path = os.path.realpath(__file__)
    path = path[: -len("test_hops_system.py")] + "/known_param_w.npy"
    known_param_w = np.load(path)
    assert np.allclose(param_w, known_param_w)

    list_state_indices_by_hmode = HS.param["LIST_STATE_INDICES_BY_HMODE"]
    known_state_indices_by_hmode = [[0], [0], [1], [1], [2], [2], [3], [3]]
    assert np.array_equal(list_state_indices_by_hmode, known_state_indices_by_hmode)

    n_l2 = HS.param["N_L2"]
    known_n_l2 = 4
    assert n_l2 == known_n_l2

    list_state_indices_by_index_L2 = HS.param["LIST_STATE_INDICES_BY_INDEX_L2"]
    known_state_indices_by_index_L2 = [[0], [1], [2], [3]]
    assert np.array_equal(
        list_state_indices_by_index_L2, known_state_indices_by_index_L2
    )

    list_index_l2_by_nmode1 = HS.param["LIST_INDEX_L2_BY_NMODE1"]
    known_index_l2_by_nmode1 = [0, 0, 1, 1, 2, 2, 3, 3]
    assert np.array_equal(list_index_l2_by_nmode1, known_index_l2_by_nmode1)


def test_initialize_true():
    """
    This function test whether initialize is creating an accurate state list when the
    calculation is adaptive
    """
    psi = np.array([0, 0, 1, 0])
    HS.initialize(True, psi)
    state_list = HS.state_list
    known_state_list = [2]
    assert state_list == known_state_list


def test_initialize_false():
    """
    This function test whether initialize is creating an accurate state list when the
    calculation is non-adaptive
    """
    psi = np.array([0, 1, 0, 0])
    HS.initialize(False, psi)
    state = HS.state_list
    known_state = [0, 1, 2, 3]
    assert np.array_equal(state, known_state)


def test_array_to_tuple():
    """ This function test whether an array properly turns into a tuple"""
    array = lop_list[0]
    array_to_tuple = HS._array_to_tuple(array)
    known_tuple = (tuple([0]), tuple([0]))
    assert array_to_tuple == known_tuple


def test_get_states_from_L2_try():
    """
    This function test to make sure _get_states_from_L2 is properly listing the states
    that the L operators interacts with
    """
    lop = lop_list[2]
    lop = HS._get_states_from_L2(lop)
    known_state_tuple = tuple([1])
    assert lop == known_state_tuple


def test_get_states_from_L2_except():
    """
    This function test to make sure _get_states_from_L2 is properly listing the states
    that the L operators interacts with
    """
    lop = sp.sparse.coo_matrix(lop_list[2])
    lop = HS._get_states_from_L2(lop)
    known_state_tuple = tuple([1])
    assert lop == known_state_tuple


def test_state_list_setter():
    """
    This test is to make sure state_list setter is working properly
    """
    # test to make sure state list is sorted
    HS.state_list = [3, 0, 1]
    state_list = HS.state_list
    known_sorted_list = [0, 1, 3]
    assert np.array_equal(state_list, known_sorted_list)

    # test to make sure the absolute indexing of L2 is working
    list_absindex_L2 = HS._list_absindex_L2
    known_absindex_L2 = [0, 1, 3]
    assert np.array_equal(list_absindex_L2, known_absindex_L2)

    # test to make sure the absolute indexing of the modes is working
    list_absindex_mode = HS._list_absindex_mode
    known_absindex_mode = [0, 1, 2, 3, 6, 7]
    assert np.array_equal(list_absindex_mode, known_absindex_mode)

    # test to make sure the sub-setting of the hamiltonian is working
    hamiltonian = HS._hamiltonian
    known_h = np.zeros([3, 3])
    known_h[0, 1] = 40
    known_h[1, 0] = 40
    assert np.array_equal(hamiltonian, np.array(known_h, dtype=np.complex128))

    # test to make sure correct number of modes are saved (relative basis)
    n_hmodes = HS._n_hmodes
    known_len = 6
    assert n_hmodes == known_len

    # test to make sure correct g is being sub-indexed (relative basis)
    g = HS._g
    known_g = np.array(HS.param["G"])[known_absindex_mode]
    assert np.array_equal(g, known_g)

    # test to make sure correct w is being sub-indexed (relative basis)
    w = HS._w
    known_w = np.array(HS.param["W"])[known_absindex_mode]
    assert np.array_equal(w, known_w)

    # test to make sure correct number of L2 operators are showing up (relative basis)
    n_l2 = HS._n_l2
    known_n_l2 = 3
    assert n_l2 == known_n_l2

    # test to make sure list_L2_coo is in the proper relative basis order
    list_L2_coo = HS._list_L2_coo[2]
    list_L2_coo = list_L2_coo.todense()
    known_L2 = np.zeros((3, 3))
    known_L2[2, 2] = 1
    assert np.array_equal(list_L2_coo, known_L2)

    # test to make sure correct relative indexing of L2 by modes is stored
    list_index_L2_by_hmode = HS._list_index_L2_by_hmode
    known_index_L2_by_hmode = [0, 0, 1, 1, 2, 2]
    assert np.array_equal(list_index_L2_by_hmode, known_index_L2_by_hmode)

    # test to make sure correct relative indexing of state indices by mode is saved
    list_state_indices_by_hmode = HS._list_state_indices_by_hmode
    known_state_indices_by_hmode = [[0], [0], [1], [1], [2], [2]]
    assert np.array_equal(list_state_indices_by_hmode, known_state_indices_by_hmode)

    # test to make sure correct relative state indices are ordered in the correct
    # relative L2 order
    list_state_indices_by_index_L2 = HS._list_state_indices_by_index_L2
    known_state_indices_by_index_L2 = [[0], [1], [2]]
    assert np.array_equal(
        list_state_indices_by_index_L2, known_state_indices_by_index_L2
    )


def test_reduce_sparse_matrix():
    """
    This function test to make sure reduced_sparse_matrix is properly taking in a
    sparse matrix and list which represents the absolute state and converting it to a
    new relative state represented in a sparse matrix
    """
    state_list = [0, 1, 3]
    full_matrix = lop_list[6]
    coo_matrix = sp.sparse.coo_matrix(full_matrix)
    coo_matrix = HS.reduce_sparse_matrix(coo_matrix, state_list)
    coo_matrix = coo_matrix.todense()
    known_matrix = np.zeros((3, 3))
    known_matrix[2, 2] = 1
    assert np.array_equal(coo_matrix, np.array(known_matrix))
