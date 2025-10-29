import os
import pytest
import numpy as np
import scipy as sp
from mesohops.basis.hops_system import HopsSystem as HSystem
from mesohops.basis.system_functions import initialize_system_dict
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from .utils import compare_dictionaries

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
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

lop_list_peierls = []
gw_sysbath_peierls = []
for i in range(nsite-1):
    l_op_peierls = np.zeros([nsite, nsite])
    l_op_peierls[i, i+1] = 1.0
    l_op_peierls[i+1, i] = 1.0
    gw_sysbath_peierls.append([-1j * np.imag(g_0), 500.0])
    lop_list_peierls.append(l_op_peierls)

sys_param_peierls = {
    "HAMILTONIAN": np.array(hs, dtype=np.complex128),
    "GW_SYSBATH": gw_sysbath_peierls,
    "L_HIER": lop_list_peierls,
    "L_NOISE1": lop_list_peierls,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": gw_sysbath_peierls,
}
HS_peierls = HSystem(sys_param_peierls)


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


def test_state_list_setter():
    """
    Tests that the state list setter correctly manages helper objects for indexing
    states for various purposes.
    """
    # test to make sure state list is sorted
    HS.state_list = [3, 0, 1]
    state_list = HS.state_list
    known_sorted_list = [0, 1, 3]
    assert np.array_equal(state_list, known_sorted_list)
    # test to make sure the sub-setting of the hamiltonian is working
    hamiltonian = HS._hamiltonian
    known_h = np.zeros([3, 3])
    known_h[0, 1] = 40
    known_h[1, 0] = 40
    assert np.array_equal(hamiltonian, np.array(known_h, dtype=np.complex128))
    # test boundary states
    HS.state_list = [1,3]
    assert HS.list_boundary_state == [0,2]
    HS.state_list = [0]
    assert HS.list_boundary_state == [1]
    HS.state_list = [3]
    assert HS.list_boundary_state == [2]
    # more complicated boundary example hamiltonian
    nsite = 6
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)
    loperator = np.zeros([nsite, nsite, nsite], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        #Add some off-diagonal Peierls terms to test list_sc
        if i > 0:
            loperator[i, i, i-1] = 1.0
            loperator[i, i-1, i] = 1.0
        if i < nsite-1:
            loperator[i, i, i+1] = 1.0
            loperator[i, i+1, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])
    hs = np.zeros([nsite, nsite])
    hs[0, 5] = 7429038
    hs[1, 3] = 80953
    hs[1, 4] = 2304985
    hs[2, 1] = 984732
    hs[2, 0] = 23478569
    hs[3, 2] = 2309857
    hs[4, 2] = 2963784
    hs[5, 0] = 98270394287
    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }
    HS2 = HSystem(sys_param)
    HS2.state_list = [1,3]
    assert HS2.list_boundary_state == [2,4]
    assert HS2.list_sc == [0,2,4]
    HS2.state_list = [0]
    assert HS2.list_boundary_state == [5]
    assert HS2.list_sc == [1,5]
    HS2.state_list = [3]
    assert HS2.list_boundary_state == [2]
    assert HS2.list_sc == [2,4]
    HS2.state_list = [2,3,4]
    assert HS2.list_boundary_state == [0,1]
    assert HS2.list_sc == [0,1,5]
    # test list_absindex_state_modes
    # test 1: One particle, two modes per site
    HS.state_list = [1,3]
    known_list_absindex_state_modes = np.array([2,3,6,7])
    list_absindex_state_modes = HS.list_absindex_state_modes
    known_list_absindex_L2_active = np.array([1,3])
    list_absindex_L2_active = HS.list_absindex_L2_active
    assert np.array_equal(known_list_absindex_state_modes, list_absindex_state_modes)
    assert np.array_equal(known_list_absindex_L2_active, list_absindex_L2_active)
    # test 2: Two particle, indistinguishable, two modes per site
    # Two-particle states given the ordering
    # (a,b) < (c,d) (if a < c) or (if a = c and b < d)
    nsite = 4
    nstate = 10
    loperator0 = np.diag([1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])
    loperator1 = np.diag([0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0])
    loperator2 = np.diag([0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0])
    loperator3 = np.diag([0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0])
    list_loperator = [loperator0,loperator1,loperator2,loperator3]
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)
    gw_sysbath = []
    lop_list = []
    for loperatori in list_loperator:
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperatori))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperatori)
    hs = np.zeros([nstate, nstate])
    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }
    HS2P = HSystem(sys_param)
    #Test 2a: just state 0
    HS2P.state_list = [0]
    known_list_absindex_state_modes = [0,1]
    list_absindex_state_modes = HS2P.list_absindex_state_modes
    known_list_absindex_L2_active = [0]
    list_absindex_L2_active = HS2P.list_absindex_L2_active
    assert np.array_equal(known_list_absindex_state_modes, list_absindex_state_modes)
    assert np.array_equal(known_list_absindex_L2_active, list_absindex_L2_active)
    #Test 2b: state 0,2
    HS2P.state_list = [0,2]
    known_list_absindex_state_modes = [0,1,4,5]
    list_absindex_state_modes = HS2P.list_absindex_state_modes
    known_list_absindex_L2_active = [0,2]
    list_absindex_L2_active = HS2P.list_absindex_L2_active
    assert np.array_equal(known_list_absindex_state_modes, list_absindex_state_modes)
    assert np.array_equal(known_list_absindex_L2_active, list_absindex_L2_active)
    #Test 2c: state 1,7,9
    HS2P.state_list = [1,7]
    known_list_absindex_state_modes = [0,1,2,3,4,5]
    list_absindex_state_modes = HS2P.list_absindex_state_modes
    known_list_absindex_L2_active = [0,1,2]
    list_absindex_L2_active = HS2P.list_absindex_L2_active
    assert np.array_equal(known_list_absindex_state_modes, list_absindex_state_modes)
    assert np.array_equal(known_list_absindex_L2_active, list_absindex_L2_active)

def test_list_destination_state():
    """
    Tests that the list of destination states - those that can receive flux from
    states in the basis - is properly constructed and managed.
    """
    # Tests that the full list of destination states for the absolute-indexed states
    # is correct.
    list_dest_by_state_ref = [[1], [0,2], [1,3], [2]]
    list_dest_by_state = HS_peierls.param["LIST_DESTINATION_STATES_BY_STATE_INDEX"]
    assert list_dest_by_state == list_dest_by_state_ref

    # Tests that the full destination state list is the full list of states,
    # sorted properly.
    HS_peierls.state_list = [0, 1, 2, 3]
    np.testing.assert_allclose(HS_peierls.list_destination_state, np.array([0, 1, 2,
                                                                            3]))
    
    # Tests the destination state list when destination states are not the source
    # states (Peierls couplings).
    HS_peierls.state_list = [0, 3]
    np.testing.assert_allclose(HS_peierls.list_destination_state, np.array([1, 2]))

    # Tests the destination state list when destination states are the source states
    # (Holstein couplings).
    HS.state_list = [0, 3]
    np.testing.assert_allclose(HS.list_destination_state, np.array([0, 3]))

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
    #Begin 2-particle tests
    #Test arbitrary diagonal matrix
    state_list = [0,1,5,6]
    full_matrix = np.zeros((7,7))
    full_matrix[0,0] = 1
    full_matrix[1,1] = 2
    full_matrix[4,4] = 1
    full_matrix[5,5] = 4
    coo_matrix = sp.sparse.coo_matrix(full_matrix)
    coo_matrix = HS.reduce_sparse_matrix(coo_matrix, state_list)
    coo_matrix = coo_matrix.todense()
    known_matrix = np.zeros((4,4))
    known_matrix[0,0] = 1
    known_matrix[1,1] = 2
    known_matrix[2,2] = 4
    known_matrix[3,3] = 0
    assert np.array_equal(coo_matrix, np.array(known_matrix))
    #Test arbitrary matrix
    state_list = [1,3,5,6]
    full_matrix = np.zeros((7,7))
    full_matrix[0,0] = 1
    full_matrix[1,1] = 2
    full_matrix[4,4] = 1
    full_matrix[5,5] = 4
    full_matrix[1,2] = 3
    full_matrix[1,3] = 5
    full_matrix[1,5] = 6
    full_matrix[3,5] = 8
    full_matrix[3,4] = 9
    full_matrix[5,6] = -3
    coo_matrix = sp.sparse.coo_matrix(full_matrix)
    coo_matrix = HS.reduce_sparse_matrix(coo_matrix, state_list)
    coo_matrix = coo_matrix.todense()
    known_matrix = np.zeros((4,4))
    known_matrix[0,0] = 2
    known_matrix[1,1] = 0
    known_matrix[2,2] = 4
    known_matrix[3,3] = 0
    known_matrix[0,1] = 5
    known_matrix[0,2] = 6
    known_matrix[1,2] = 8
    known_matrix[2,3] = -3
    assert np.array_equal(coo_matrix, np.array(known_matrix))

def test_dict_relative_index_by_state():
    HS.state_list = [3, 0, 2]
    # Note that the sorted state list is [0, 2, 3]
    assert len(HS.dict_relative_index_by_state.items()) == len(HS.state_list)
    assert HS.dict_relative_index_by_state[0] == 0
    assert HS.dict_relative_index_by_state[2] == 1
    assert HS.dict_relative_index_by_state[3] == 2


def test_from_file_missing():
    """Ensures that constructing with a missing file raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        hs_loaded = HSystem("missing.pkl")


def test_save_and_load_full(tmp_path):
    """Comprehensively tests saving and loading of system parameters."""
    # Small two state system used to keep the test light weight
    ham = np.zeros((2, 2))
    l1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    l2 = np.array([[0.0, 0.0], [0.0, 1.0]])
    l3 = np.array([[0.0, 1.0], [1.0, 0.0]])

    param = {
        "HAMILTONIAN": ham,
        "GW_SYSBATH": [(0.1, 1.0), (0.2, 1.1), (0.3, 1.2)],
        "L_HIER": [l1, l2, l3],
        "L_NOISE1": [l1, l2, l3],
        "PARAM_NOISE1": [(0.1, 1.0), (0.2, 1.1), (0.3, 1.2)],
        "ALPHA_NOISE1": bcf_exp,
    }

    hs = HSystem(param)

    fname = tmp_path / "sys.pkl"
    hs.save_dict_param(fname)
    assert fname.exists()

    hs_loaded = HSystem(fname)

    # Ensure all keys are present and values are equal
    compare_dictionaries(hs.param, hs_loaded.param)


def test_load_invalid_type():
    """Ensures that constructing with an invalid type raises a TypeError."""
    with pytest.raises(TypeError):
        HSystem(123)
