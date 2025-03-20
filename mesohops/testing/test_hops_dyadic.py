import numpy as np
import pytest
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp,  \
    ishizaki_decomposition_bcf_dl
from mesohops.util.physical_constants import hbar, kB
from mesohops.dynamics.hops_dyadic import DyadicTrajectory as DHOPS
from scipy import sparse
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.spectroscopy_analysis import _response_function_calc

noise_param = {
    "SEED": 10,
    "MODEL": "FFT_FILTER",
    "TLEN": 50.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

nsite=5

list_lop_dense= []
list_lop_sparse= []
for i in range(nsite):
    lop_dense=np.zeros((nsite+1,nsite+1))
    lop_dense[i+1, i+1] = 1.0
    list_lop_dense.append(lop_dense)
    list_lop_sparse.append(sparse.coo_matrix(lop_dense))

V = 10
H_ex = (np.diag([0]*nsite)
          + np.diag([V] * (nsite - 1), k=-1)
          + np.diag([V] * (nsite - 1), k=1))
H_sys_dense=np.zeros((nsite+1,nsite+1))
H_sys_dense[1:,1:]=H_ex

H_sys_sparse=sparse.coo_matrix(H_sys_dense)

sys_param_dense = {
    "HAMILTONIAN": H_sys_dense,
    "GW_SYSBATH": [[10.0, 10.0]]*5,
    "L_HIER": list_lop_dense,
    "L_NOISE1": list_lop_dense*2,
    "L_LT_CORR":list_lop_dense,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0]]*10,
    'PARAM_LT_CORR':[0]*5
}

sys_param_sparse = {
    "HAMILTONIAN": H_sys_sparse,
    "GW_SYSBATH": [[10.0, 10.0]]*5,
    "L_HIER": list_lop_sparse,
    "L_NOISE1": list_lop_sparse*2,
    "L_LT_CORR":list_lop_sparse,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0]]*10,
    'PARAM_LT_CORR':[0]*5
}

hier_param = {"MAXHIER": 5}

eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {
    "INTEGRATOR": "RUNGE_KUTTA",
    'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
    'EARLY_INTEGRATOR_STEPS': 5,
    'INCHWORM_CAP': 5,
    'STATIC_BASIS': None
}

dhops_dense = DHOPS(
    sys_param_dense,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
)
dhops_sparse = DHOPS(
    sys_param_sparse,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
)

t_max = 10.0
t_step = 2.0
psi_k = np.zeros(nsite+1, dtype=np.complex64)
psi_k[0]=1
psi_b = np.zeros(nsite+1, dtype=np.complex64)
psi_b[0]=1
dhops_sparse.make_adaptive(0.0000001,0)
dhops_dense.initialize(psi_k, psi_b)
dhops_sparse.initialize(psi_k, psi_b)

Op_ket_dense=np.zeros([nsite + 1, nsite + 1])
Op_ket_dense[1:,0]=1
Op_bra_dense=np.zeros([nsite + 1, nsite + 1])
Op_bra_dense[3:,0]=1

# Helper function
# ---------------

def make_op_dyadic(op_hilbert,side):
    op_dim = np.shape(op_hilbert)[0]
    op = np.zeros((2 * op_dim, 2 * op_dim))
    op[np.arange(2*op_dim), np.arange(2*op_dim)] = 1
    if side == 'bra':
        op[op_dim:, op_dim:] = op_hilbert
    elif side == 'ket':
        op[:op_dim, :op_dim] = op_hilbert
    return(op)

@pytest.mark.order(1)
def test_dyad_initialization():
    """
    Tests the dyadic hops trajectory initialize function.
    """
    # Dense or Sparse
    # ---------------
    psi_ref = np.concatenate((psi_k, psi_b))
    psi_ref = psi_ref/ np.sqrt(np.sum(np.abs(psi_ref) ** 2))
    np.testing.assert_allclose(dhops_dense.psi, psi_ref)

    assert len(dhops_dense.list_response_norm_sq)==1

@pytest.mark.order(2)
def test_M2_dyad_conversion():
    """
    Tests the M2_dyad_conversion function, which converts a given matrix M into a
    block-diagonal matrix of the form [[M, 0],[0,M]], in both dense and sparse formats.
    """

    # Dense Construct
    # ----------------
    Hamiltonian_ref = np.zeros([2*(nsite+1),2*(nsite+1)], dtype=np.float64)
    Hamiltonian_ref[nsite+1:, nsite+1:] = H_sys_dense
    Hamiltonian_ref[:nsite+1, :nsite+1] = H_sys_dense
    np.testing.assert_allclose(sys_param_dense['HAMILTONIAN'], Hamiltonian_ref)

    list_lop_dense_ref =  np.zeros([nsite, 2*(nsite+1), 2*(nsite+1)], dtype=np.float64)
    for i in range(nsite):
        list_lop_dense_ref[i, nsite+1:, nsite+1:] = list_lop_dense[i]
        list_lop_dense_ref[i, :nsite+1, :nsite+1] = list_lop_dense[i]

    np.testing.assert_allclose(sys_param_dense['L_HIER'], list_lop_dense_ref)
    np.testing.assert_allclose(sys_param_dense['L_NOISE1'], list(list_lop_dense_ref)*2)
    np.testing.assert_allclose(sys_param_dense['L_LT_CORR'], list_lop_dense_ref)

    # Sparse Construct
    # ----------------

    Hamiltonian_sparse_ref = sparse.coo_matrix(Hamiltonian_ref)

    assert  (sys_param_sparse['HAMILTONIAN'] != Hamiltonian_sparse_ref).nnz == 0

    list_lop_sparse_ref = []
    for i in range(nsite):
        list_lop_sparse_ref.append(sparse.coo_matrix(list_lop_dense_ref[i]))

    for i, (matrix_test, matrix_ref) in enumerate(zip(sys_param_sparse["L_HIER"], list_lop_sparse_ref)):
        assert (matrix_test != matrix_ref).nnz == 0
    for i, (matrix_test, matrix_ref) in enumerate(zip(sys_param_sparse["L_NOISE1"], list_lop_sparse_ref*2)):
        assert (matrix_test != matrix_ref).nnz == 0
    for i, (matrix_test, matrix_ref) in enumerate(zip(sys_param_sparse["L_LT_CORR"], list_lop_sparse_ref)):
        assert (matrix_test != matrix_ref).nnz == 0

@pytest.mark.order(3)
def test_dyad_operator():
    """
    Tests the wavefunction obtained after a ket or a bra operation using _dyad_operator
    in hops_dyadic.py.
    """

    # Dense Construct
    # ----------------
    dhops_dense._dyad_operator(Op_ket_dense, 'ket')
    psi_1_ref = np.concatenate((Op_ket_dense@psi_k, psi_b))/np.sqrt(np.sum(np.abs(
        np.concatenate((Op_ket_dense@psi_k, psi_b))) ** 2))
    np.testing.assert_allclose(dhops_dense.psi, psi_1_ref)

    dhops_dense._dyad_operator(Op_bra_dense, 'bra')
    psi_2_ref = np.concatenate((Op_ket_dense @ psi_k, Op_bra_dense @ psi_b)) / np.sqrt(
        np.sum(np.abs(np.concatenate((Op_ket_dense @ psi_k,
                                      Op_bra_dense @ psi_b))) ** 2))
    np.testing.assert_allclose(dhops_dense.psi, psi_2_ref)

    try:
        dhops_dense._dyad_operator(Op_bra_dense, 'braket')
    except UnsupportedRequest as excinfo:
        if 'sides other than "ket" or "bra"' not in str(excinfo):
             pytest.fail()

    # Sparse Construct
    # ----------------

    Op_ket_sparse = sparse.coo_matrix(Op_ket_dense)
    Op_bra_sparse = sparse.coo_matrix(Op_bra_dense)

    dhops_sparse._dyad_operator(Op_ket_sparse, 'ket')
    psi_1_ref = np.concatenate((Op_ket_sparse@psi_k, psi_b))
    psi_1_ref = psi_1_ref/np.sqrt(np.sum(np.abs(psi_1_ref) ** 2))
    np.testing.assert_allclose(dhops_sparse.psi, psi_1_ref)

    dhops_sparse._dyad_operator(Op_bra_sparse, 'bra')
    psi_2_ref = np.concatenate((Op_ket_sparse@psi_k, Op_bra_sparse@psi_b))/\
                np.sqrt(np.sum(np.abs(np.concatenate((Op_ket_sparse@psi_k,
                                                      Op_bra_sparse@psi_b))) ** 2))
    np.testing.assert_allclose(dhops_sparse.psi, psi_2_ref)

    try:
        dhops_sparse._dyad_operator(Op_bra_sparse, 'braket')
    except UnsupportedRequest as excinfo:
        if 'sides other than "ket" or "bra"' not in str(excinfo):
             pytest.fail()

@pytest.mark.order(4)
def test_norm_comp_list():
    """
    Tests the list of normalization correction factors after mutiple operations.
    """
    psi_0 = np.concatenate((psi_k, psi_b))
    list_norm_factor_ref=[np.linalg.norm(psi_0) ** 2]
    Op_bra_2 = np.zeros([nsite + 1, nsite + 1])
    Op_bra_2[0, 1:] = 1
    dhops_dense.propagate(10, 2)
    dhops_dense._dyad_operator(Op_bra_2, 'bra')
    dhops_dense.propagate(10, 2)
    list_norm_factor_test = dhops_dense.list_response_norm_sq

    psi_traj= dhops_dense.storage['psi_traj']
    psi_init = np.concatenate((psi_k, psi_b))
    psi_init = psi_init / np.sqrt(np.sum(np.abs(psi_init) ** 2))
    # First operation on the ket state for excitation
    ket_op_dyd = make_op_dyadic(Op_ket_dense,'ket')
    psi_op_ket=ket_op_dyd @ psi_init
    list_norm_factor_ref.append(np.linalg.norm(psi_op_ket) ** 2)
    # First operation on the bra state for excitation
    bra_op_dyd = make_op_dyadic(Op_bra_dense, 'bra')
    psi_op_bra = bra_op_dyd @ (psi_op_ket/ np.sqrt(np.sum(np.abs(psi_op_ket) ** 2)))
    list_norm_factor_ref.append(np.linalg.norm(psi_op_bra) ** 2)
    # Second operation on the bra state for de-excitation
    third_op_dyd = make_op_dyadic(Op_bra_2, 'bra')
    psi_op_third = third_op_dyd @ psi_traj[5]
    list_norm_factor_ref.append(np.linalg.norm(psi_op_third) ** 2)

    np.testing.assert_allclose(list_norm_factor_test, list_norm_factor_ref)

@pytest.mark.order(5)
def test_response_function_comp():
    """
    Tests the response_function_comp function in hops_dyadic.

    NOTE: The function _response_function_comp in hops_dyadic.py is just a wrapper
    function for _response_function_calc in spectroscopy_analysis.py, therefore both are
    tested here together.
    """
    # Dense Construct
    # ----------------
    F_dense = np.zeros((2*nsite+2, 2*nsite+2))
    F_dense[nsite + 1, 1:(nsite+1)] = np.array([1] * (nsite))

    response_fn_dense_ref=[ (np.prod(dhops_dense.list_response_norm_sq) /
                             (np.linalg.norm(psi_t) ** 2))
                            * (np.conj(psi_t) @ F_dense @ psi_t)
                            for psi_t in dhops_dense.storage['psi_traj'][4:]]
    response_fn_dense_test = dhops_dense._response_function_comp(F_dense, 3)

    np.testing.assert_allclose(response_fn_dense_ref,response_fn_dense_test)

    # Sparse Construct
    # ----------------
    Op_bra_2 = np.zeros([nsite + 1, nsite + 1])
    Op_bra_2[0, 1:] = 1
    Op_bra_2_sparse = sparse.coo_matrix(Op_bra_2)
    dhops_sparse.propagate(10, 2)
    dhops_sparse._dyad_operator(Op_bra_2_sparse, 'bra')
    dhops_sparse.propagate(10, 2)
    F_sparse=sparse.csr_matrix(F_dense)

    traj_csr = sparse.csr_matrix(dhops_sparse.storage['psi_traj_sparse'])
    traj_t = traj_csr.transpose()

    response_fn_sparse_ref = np.ravel([(np.prod(dhops_sparse.list_response_norm_sq) /
            (np.linalg.norm(traj_t[:, col].data) ** 2)) * (np.conj(traj_t[:, col].T) @
        (F_sparse @ traj_t[:, col])).todense()[0] for col in range(4, traj_t.shape[1])])
    response_fn_sparse_test = dhops_sparse._response_function_comp(F_sparse, 3)

    assert np.array_equal(response_fn_sparse_ref, response_fn_sparse_test)

