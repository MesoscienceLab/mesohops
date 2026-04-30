import os
import pytest
import numpy as np
import scipy as sp
from mesohops.basis.hops_system import HopsSystem as HSystem
from mesohops.basis.system_functions import initialize_system_dict
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.physical_constants import hbar
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


# ------------------------------------------------------------
# TEST: list_dict_L2_nnz matches COO nonzero entries
# ------------------------------------------------------------
def test_list_dict_l2_nnz_construction():
    """Tests that list_dict_L2_nnz matches nonzero entries in LIST_L2_COO."""
    list_l2_coo = HS.param["LIST_L2_COO"]
    list_dict_l2_nnz = HS.param["list_dict_L2_nnz"]

    # One nnz dict per L2 operator
    assert len(list_dict_l2_nnz) == len(list_l2_coo)

    for idx_l2, l2_coo in enumerate(list_l2_coo):
        # Build a reference dict from COO triplets: (row, col) -> data
        dict_ref = {}
        for row, col, data in zip(l2_coo.row, l2_coo.col, l2_coo.data):
            dict_ref[(row, col)] = data

        # Verify the nnz dict has the same keys and values as the reference
        assert set(list_dict_l2_nnz[idx_l2].keys()) == set(dict_ref.keys())
        for key in dict_ref:
            assert np.allclose(list_dict_l2_nnz[idx_l2][key], dict_ref[key])


# ------------------------------------------------------------
# TEST: list_L2_off_diag flags match COO structure
# ------------------------------------------------------------
def test_list_l2_off_diag_consistency():
    """Tests that list_L2_off_diag is consistent with LIST_L2_COO structure."""
    # Loop over both diagonal (HS) and off-diagonal (HS_peierls) L-operators
    for hs_obj in [HS, HS_peierls]:
        list_l2_off_diag = hs_obj.param["list_L2_off_diag"]
        list_l2_coo = hs_obj.param["LIST_L2_COO"]

        # One flag per L2 operator
        assert len(list_l2_off_diag) == len(list_l2_coo)
        for idx_l2, l2_coo in enumerate(list_l2_coo):
            # True if any COO entry has row != col, False if purely diagonal
            expected_flag = not np.allclose(l2_coo.col, l2_coo.row)
            assert list_l2_off_diag[idx_l2] == expected_flag


def _initialize_minimal_system_with_l2(l2_operator):
    """Helper to initialize system params for list_dict_L2_nnz construction tests."""
    return initialize_system_dict(
        {
            "HAMILTONIAN": np.zeros((3, 3), dtype=np.complex128),
            "GW_SYSBATH": [(0.1 + 0.0j, 1.0)],
            "L_HIER": [l2_operator],
            "L_NOISE1": [l2_operator],
            "PARAM_NOISE1": [(0.1 + 0.0j, 1.0)],
            "ALPHA_NOISE1": bcf_exp,
        }
    )


# ------------------------------------------------------------
# TEST: Duplicate real COO coordinates are accumulated
# ------------------------------------------------------------
def test_list_dict_l2_nnz_accumulates_duplicate_real_coordinates():
    """Duplicate real-valued COO coordinates should be accumulated in the dict."""
    # COO input: (0,1) appears twice with values 1.25 and -0.25, (1,2) once
    l2_op = sp.sparse.coo_matrix(
        ([1.25, -0.25, 2.0], ([0, 0, 1], [1, 1, 2])), shape=(3, 3)
    )
    param = _initialize_minimal_system_with_l2(l2_op)
    dict_l2_nnz = param["list_dict_L2_nnz"][0]

    # Two unique coordinates after accumulation
    assert len(dict_l2_nnz) == 2
    # (0,1): 1.25 + (-0.25) = 1.0
    assert np.allclose(dict_l2_nnz[(0, 1)], 1.0)
    # (1,2): single entry, unchanged
    assert np.allclose(dict_l2_nnz[(1, 2)], 2.0)


# ------------------------------------------------------------
# TEST: Duplicate complex COO coordinates are accumulated
# ------------------------------------------------------------
def test_list_dict_l2_nnz_accumulates_duplicate_complex_coordinates():
    """Duplicate complex-valued COO coordinates should be accumulated in the dict."""
    # COO input: (0,1) appears twice with values (1+2j) and (3-0.5j), (2,2) once
    l2_op = sp.sparse.coo_matrix(
        ([1 + 2j, 3 - 0.5j, -1j], ([0, 0, 2], [1, 1, 2])), shape=(3, 3)
    )
    param = _initialize_minimal_system_with_l2(l2_op)
    dict_l2_nnz = param["list_dict_L2_nnz"][0]

    # Two unique coordinates after accumulation
    assert len(dict_l2_nnz) == 2
    # (0,1): (1+2j) + (3-0.5j) = 4+1.5j
    assert np.allclose(dict_l2_nnz[(0, 1)], 4 + 1.5j)
    # (2,2): single diagonal entry, unchanged
    assert np.allclose(dict_l2_nnz[(2, 2)], -1j)



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
    assert HS.list_bndstateidx_abs == [0,2]
    HS.state_list = [0]
    assert HS.list_bndstateidx_abs == [1]
    HS.state_list = [3]
    assert HS.list_bndstateidx_abs == [2]
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
        #Add some off-diagonal Peierls terms to test list_fullbndidx_abs
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
    hs[2, 1] = -100000
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
    assert HS2.list_bndstateidx_abs == [2,4]
    assert HS2.list_fullbndidx_abs == [0,2,4]
    HS2.state_list = [0]
    assert HS2.list_bndstateidx_abs == [5]
    assert HS2.list_fullbndidx_abs == [1,5]
    HS2.state_list = [3]
    assert HS2.list_bndstateidx_abs == [2]
    assert HS2.list_fullbndidx_abs == [2,4]
    HS2.state_list = [2,3,4]
    assert HS2.list_bndstateidx_abs == [0,1]
    assert HS2.list_fullbndidx_abs == [0,1,5]
    # Check system timescale
    HS2.state_list = [0,5]
    assert np.allclose(HS2.system_timescale, hbar/98270394287)
    HS2.state_list = [1,2,3]
    assert np.allclose(HS2.system_timescale, hbar/(2309857+100000))
    # test list_statemodeidx_abs
    # test 1: One particle, two modes per site
    HS.state_list = [1,3]
    known_list_statemodeidx_abs = np.array([2,3,6,7])
    list_statemodeidx_abs = HS.list_statemodeidx_abs
    known_list_activel2idx_abs = np.array([1,3])
    list_activel2idx_abs = HS.list_activel2idx_abs
    assert np.array_equal(known_list_statemodeidx_abs, list_statemodeidx_abs)
    assert np.array_equal(known_list_activel2idx_abs, list_activel2idx_abs)
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
    known_list_statemodeidx_abs = [0,1]
    list_statemodeidx_abs = HS2P.list_statemodeidx_abs
    known_list_activel2idx_abs = [0]
    list_activel2idx_abs = HS2P.list_activel2idx_abs
    assert np.array_equal(known_list_statemodeidx_abs, list_statemodeidx_abs)
    assert np.array_equal(known_list_activel2idx_abs, list_activel2idx_abs)
    #Test 2b: state 0,2
    HS2P.state_list = [0,2]
    known_list_statemodeidx_abs = [0,1,4,5]
    list_statemodeidx_abs = HS2P.list_statemodeidx_abs
    known_list_activel2idx_abs = [0,2]
    list_activel2idx_abs = HS2P.list_activel2idx_abs
    assert np.array_equal(known_list_statemodeidx_abs, list_statemodeidx_abs)
    assert np.array_equal(known_list_activel2idx_abs, list_activel2idx_abs)
    #Test 2c: state 1,7,9
    HS2P.state_list = [1,7]
    known_list_statemodeidx_abs = [0,1,2,3,4,5]
    list_statemodeidx_abs = HS2P.list_statemodeidx_abs
    known_list_activel2idx_abs = [0,1,2]
    list_activel2idx_abs = HS2P.list_activel2idx_abs
    assert np.array_equal(known_list_statemodeidx_abs, list_statemodeidx_abs)
    assert np.array_equal(known_list_activel2idx_abs, list_activel2idx_abs)

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


def test_reduce_sparse_matrix_non_peierls():
    """Tests diagonal-only reduction for non-Peierls operators."""
    dict_l2_nnz = {
        (1, 1): 10.0,
        (3, 3): 20.0,
        (1, 3): 99.0,  # should be ignored when off_diag is False
    }
    state_list = [1, 3, 4]
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_l2_nnz, state_list, False
    )
    known_matrix = np.zeros((3, 3))
    known_matrix[0, 0] = 10.0
    known_matrix[1, 1] = 20.0
    assert np.array_equal(coo_matrix.todense(), known_matrix)


def test_reduce_sparse_matrix_peierls():
    """Tests full pairwise reduction for Peierls operators."""
    dict_l2_nnz = {
        (1, 1): 2.0,
        (1, 4): 3.0,
        (4, 1): 5.0,
        (4, 4): 7.0,
        (10, 10): 11.0,  # outside selected states
    }
    state_list = [4, 1]
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_l2_nnz, state_list, True
    )
    known_matrix = np.zeros((2, 2))
    known_matrix[0, 0] = 7.0
    known_matrix[0, 1] = 5.0
    known_matrix[1, 0] = 3.0
    known_matrix[1, 1] = 2.0
    assert np.array_equal(coo_matrix.todense(), known_matrix)


def test_reduce_sparse_matrix_missing_keys_and_non_keyerror():
    """Tests missing-key behavior and non-KeyError propagation."""
    state_list = [1, 2]
    coo_matrix = HSystem.reduce_sparse_matrix({(1, 1): 5.0},
                                                             state_list, False)
    known_matrix = np.zeros((2, 2))
    known_matrix[0, 0] = 5.0
    assert np.array_equal(coo_matrix.todense(), known_matrix)

    class BrokenDict:
        def __getitem__(self, key):
            raise TypeError("invalid backend type")

    with pytest.raises(TypeError):
        HSystem.reduce_sparse_matrix(BrokenDict(),
                                                    [0, 1], False)


def test_extended_basis_outputs_consistency():
    """
    Tests that extended basis indices and extended Hamiltonian are self-consistent.
    """
    HS.state_list = [1, 3]
    ext_size = HS.H2_hamiltonian_extd.shape[0]
    list_state_extd = [None] * ext_size

    for idx, state in zip(HS.list_stateidx_extd, HS.state_list):
        list_state_extd[idx] = state
    for idx, state in zip(HS.list_bndstateidx_extd, HS.list_fullbndidx_abs):
        list_state_extd[idx] = state

    assert all(state is not None for state in list_state_extd)
    assert set(list_state_extd) == set(HS.state_list) | set(HS.list_fullbndidx_abs)
    assert set(HS.list_stateidx_extd).isdisjoint(set(HS.list_bndstateidx_extd))

    known_hamiltonian_ext = HS.param["SPARSE_HAMILTONIAN"][
        np.ix_(list_state_extd, list_state_extd)
    ]
    np.testing.assert_allclose(HS.H2_hamiltonian_extd.todense(),
                               known_hamiltonian_ext.todense())


def test_hamiltonian_ext_matches_direct_sparse_slice_multiple_subsets():
    """Tests H2_hamiltonian_extd against direct sparse slicing for multiple subsets."""
    test_state_lists = [[0], [1, 3], [0, 2, 3]]
    for state_list in test_state_lists:
        HS.state_list = state_list
        ext_size = HS.H2_hamiltonian_extd.shape[0]
        list_state_extd = [None] * ext_size

        for idx, state in zip(HS.list_stateidx_extd, HS.state_list):
            list_state_extd[idx] = state
        for idx, state in zip(HS.list_bndstateidx_extd, HS.list_fullbndidx_abs):
            list_state_extd[idx] = state

        known_hamiltonian_ext = HS.param["SPARSE_HAMILTONIAN"][
            np.ix_(list_state_extd, list_state_extd)
        ]
        np.testing.assert_allclose(
            HS.H2_hamiltonian_extd.todense(), known_hamiltonian_ext.todense()
        )


def test_extended_basis_indexing_is_deterministic():
    """Tests that repeated assignment gives stable extended indexing."""
    HS.state_list = [1, 3]
    list_basis_index_ext_ref = list(HS.list_stateidx_extd)
    list_boundary_index_ext_ref = list(HS.list_bndstateidx_extd)
    hamiltonian_ext_ref = HS.H2_hamiltonian_extd.todense().copy()

    HS.state_list = [1, 3]
    assert list(HS.list_stateidx_extd) == list_basis_index_ext_ref
    assert list(HS.list_bndstateidx_extd) == list_boundary_index_ext_ref
    np.testing.assert_allclose(HS.H2_hamiltonian_extd.todense(), hamiltonian_ext_ref)


def test_reduce_sparse_matrix_size_invariant_off_diag_edge_cases():
    """Tests off-diagonal reduction edge cases."""
    state_list = [10, 20, 30]
    dict_l2_nnz = {
        (10, 20): 1.0,
        (20, 10): -2.0,
        (30, 30): 3.0,
        (999, 10): 7.0,  # outside selected states
        (20, 888): 8.0,  # outside selected states
    }
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_l2_nnz, state_list, True
    )
    known_matrix = np.zeros((3, 3))
    known_matrix[0, 1] = 1.0
    known_matrix[1, 0] = -2.0
    known_matrix[2, 2] = 3.0
    assert np.array_equal(coo_matrix.todense(), known_matrix)

    empty_matrix = HSystem.reduce_sparse_matrix({}, state_list, True)
    assert empty_matrix.shape == (3, 3)
    assert empty_matrix.nnz == 0


# ------------------------------------------------------------
# TEST: Reduced diagonal produces compact matrix
# ------------------------------------------------------------
def test_reduce_sparse_matrix_reduce_diag():
    """Tests that reduce_sparse_matrix with filter_nz=True and
    off_diag=False produces a compact matrix containing only rows/columns
    for states with nonzero entries."""
    dict_l2_nnz = {
        (1, 1): 10.0,
        (3, 3): 20.0,
    }
    state_list = [1, 2, 3, 4]

    # This case tests that the matrix excludes states without entries
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_l2_nnz, state_list, False, filter_nz=True
    )
    known_matrix = np.zeros((2, 2))
    known_matrix[0, 0] = 10.0
    known_matrix[1, 1] = 20.0
    assert coo_matrix.shape == (2, 2)
    assert np.array_equal(coo_matrix.todense(), known_matrix)

    # This case tests that an empty dict produces a 0x0 matrix
    empty_matrix = HSystem.reduce_sparse_matrix(
        {}, state_list, False, filter_nz=True
    )
    assert empty_matrix.shape == (0, 0)
    assert empty_matrix.nnz == 0


# ------------------------------------------------------------
# TEST: Reduced off-diagonal produces compact matrix
# ------------------------------------------------------------
def test_reduce_sparse_matrix_reduce_off_diag():
    """Tests that reduce_sparse_matrix with filter_nz=True and
    off_diag=True produces a compact matrix containing only states involved
    in nonzero entries."""
    dict_l2_nnz = {
        (1, 1): 2.0,
        (1, 4): 3.0,
        (4, 1): 5.0,
        (4, 4): 7.0,
    }
    state_list = [1, 2, 3, 4]

    # This case tests that states 2 and 3 are excluded from the matrix
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_l2_nnz, state_list, True, filter_nz=True
    )
    known_matrix = np.array([
        [2.0, 3.0],
        [5.0, 7.0],
    ])
    assert coo_matrix.shape == (2, 2)
    assert np.array_equal(coo_matrix.todense(), known_matrix)

    # This case tests that an empty dict produces a 0x0 matrix
    empty_matrix = HSystem.reduce_sparse_matrix(
        {}, state_list, True, filter_nz=True
    )
    assert empty_matrix.shape == (0, 0)
    assert empty_matrix.nnz == 0


# ------------------------------------------------------------
# TEST: Reduced handles partial overlap
# ------------------------------------------------------------
def test_reduce_sparse_matrix_reduce_partial():
    """Tests that reduce_sparse_matrix with filter_nz=True
    correctly handles states where only some are involved in nonzero
    entries."""
    dict_l2_nnz = {
        (10, 20): 1.0,
        (20, 10): -2.0,
        (30, 30): 3.0,
    }

    # This case tests off-diag reduced with partial overlap
    # State 40 has no entries and should be excluded
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_l2_nnz, [10, 20, 30, 40], True, filter_nz=True
    )
    # Nonzero states: [10, 20, 30]
    known_matrix = np.zeros((3, 3))
    known_matrix[0, 1] = 1.0
    known_matrix[1, 0] = -2.0
    known_matrix[2, 2] = 3.0
    assert coo_matrix.shape == (3, 3)
    assert np.array_equal(coo_matrix.todense(), known_matrix)

    # This case tests diag reduced with partial overlap
    # State 1 has no entry and should be excluded
    dict_diag = {(3, 3): 200.0, (5, 5): 100.0}
    coo_matrix = HSystem.reduce_sparse_matrix(
        dict_diag, [1, 3, 5], False, filter_nz=True
    )
    # Nonzero states: [3, 5]
    known_matrix = np.zeros((2, 2))
    known_matrix[0, 0] = 200.0
    known_matrix[1, 1] = 100.0
    assert coo_matrix.shape == (2, 2)
    assert np.array_equal(coo_matrix.todense(), known_matrix)


def test_reduce_sparse_matrix_off_diag_reduce_non_keyerror_propagates():
    """Tests non-KeyError propagation for off_diag=True, filter_nz=True path."""
    class BrokenDict:
        def __contains__(self, key):
            # Ensure filter_nz=True path includes states for later __getitem__ access.
            return True

        def __getitem__(self, key):
            raise TypeError("invalid backend type")

    with pytest.raises(TypeError):
        HSystem.reduce_sparse_matrix(BrokenDict(), [0, 1], True, filter_nz=True)
