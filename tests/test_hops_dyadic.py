import time as timer

import numpy as np
import pytest
from scipy import sparse

from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_dyadic import DyadicTrajectory as DHOPS
from mesohops.util.exceptions import UnsupportedRequest

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

sys_param_init_timing_control = {
    "HAMILTONIAN": H_sys_dense,
    "GW_SYSBATH": [[10.0, 10.0]]*5,
    "L_HIER": list_lop_dense,
    "L_NOISE1": list_lop_dense*2,
    "L_LT_CORR":list_lop_dense,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0]]*10,
    'PARAM_LT_CORR':[0]*5
}

sys_param_init_timing_plus_1sec = {
    "HAMILTONIAN": H_sys_dense,
    "GW_SYSBATH": [[10.0, 10.0]]*5,
    "L_HIER": list_lop_dense,
    "L_NOISE1": list_lop_dense*2,
    "L_LT_CORR":list_lop_dense,
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

    # Checks to make sure initialization is timed correctly by measuring control timing
    # and comparing to the time given by starting a timer, feeding in a timer_checkpoint
    # and then checking the time elapsed.

    d_hops_control = DHOPS(
        sys_param_init_timing_control,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    d_hops_control.initialize(psi_k, psi_b)

    init_time_control = d_hops_control.storage.metadata['INITIALIZATION_TIME']

    d_hops_plus_1sec = DHOPS(
        sys_param_init_timing_plus_1sec,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )

    timer_checkpoint = timer.time()

    # wait 1 second
    timer.sleep(1.0)
    d_hops_plus_1sec.initialize(psi_k, psi_b, timer_checkpoint=timer_checkpoint)
    init_time_plus_1sec = d_hops_plus_1sec.storage.metadata['INITIALIZATION_TIME']

    # checks to make sure the time is roughly one second longer than the control time
    assert np.allclose(init_time_plus_1sec-1, init_time_control, atol=1e-1)


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


    M2_complex = np.array([[1 + 2j, 2 - 1j],
                          [3 + 0j, 4 - 0.5j]], dtype=np.complex128)
    M2_dyad_ref = np.array([[1 + 2j, 2 - 1j, 0, 0],
                          [3 + 0j, 4 - 0.5j, 0, 0],[0, 0, 1 + 2j, 2 - 1j],
                          [0, 0, 3 + 0j, 4 - 0.5j]], dtype=np.complex128)

    M2_dyad_test = dhops_dense._M2_dyad_conversion(M2_complex)  # call method manually

    np.testing.assert_allclose(M2_dyad_ref, M2_dyad_test)
    # Check dtype
    assert sys_param_dense['HAMILTONIAN'].dtype == np.complex128
    assert M2_dyad_test.dtype == np.complex128

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

    with pytest.raises(UnsupportedRequest) as excinfo:
        dhops_dense._dyad_operator(Op_bra_dense, 'braket')
    assert 'sides other than "ket" or "bra"' in str(excinfo.value)

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

    with pytest.raises(UnsupportedRequest) as excinfo:
        dhops_sparse._dyad_operator(Op_bra_sparse, 'braket')
    assert 'sides other than "ket" or "bra"' in str(excinfo.value)

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
            (np.linalg.norm(traj_t[:, [col]].data) ** 2)) * (np.conj(traj_t[:, [col]].T) @
        (F_sparse @ traj_t[:, [col]])).todense()[0] for col in range(4, traj_t.shape[1])])
    response_fn_sparse_test = dhops_sparse._response_function_comp(F_sparse, 3)

    assert np.array_equal(response_fn_sparse_ref, response_fn_sparse_test)


def _build_local_dyadic_case(use_sparse_ops=False):
    """
    Builds a compact dyadic test system and staged operators for checkpoint tests.

    Parameters
    ----------
    1. use_sparse_ops : bool
                        If True, returns sparse operator matrices for operator
                        application; otherwise returns dense arrays.
    """
    nsite_local = 3
    noise_param_local = {
        "SEED": 123,
        "MODEL": "FFT_FILTER",
        "TLEN": 80.0,
        "TAU": 1.0,
    }
    eom_param_local = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    hier_param_local = {"MAXHIER": 3}
    integrator_param_local = {
        "INTEGRATOR": "RUNGE_KUTTA",
        "EARLY_ADAPTIVE_INTEGRATOR": "INCH_WORM",
        "EARLY_INTEGRATOR_STEPS": 5,
        "INCHWORM_CAP": 5,
        "STATIC_BASIS": None,
    }

    list_lop = []
    for i in range(nsite_local):
        lop = np.zeros((nsite_local + 1, nsite_local + 1), dtype=np.float64)
        lop[i + 1, i + 1] = 1.0
        list_lop.append(lop)

    V = 8.0
    H_ex = (np.diag([0.0] * nsite_local)
            + np.diag([V] * (nsite_local - 1), k=-1)
            + np.diag([V] * (nsite_local - 1), k=1))
    H_sys = np.zeros((nsite_local + 1, nsite_local + 1), dtype=np.float64)
    H_sys[1:, 1:] = H_ex

    sys_param_local = {
        "HAMILTONIAN": H_sys,
        "GW_SYSBATH": [[10.0, 10.0]] * nsite_local,
        "L_HIER": list_lop,
        "L_NOISE1": list_lop * 2,
        "L_LT_CORR": list_lop,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": [[10.0, 10.0]] * (2 * nsite_local),
        "PARAM_LT_CORR": [0.0] * nsite_local,
    }

    psi_k_local = np.zeros(nsite_local + 1, dtype=np.complex128)
    psi_k_local[0] = 1.0
    psi_b_local = np.zeros(nsite_local + 1, dtype=np.complex128)
    psi_b_local[0] = 1.0

    op_ket_exc = np.zeros((nsite_local + 1, nsite_local + 1), dtype=np.float64)
    op_ket_exc[1:, 0] = 1.0
    op_bra_exc = np.zeros((nsite_local + 1, nsite_local + 1), dtype=np.float64)
    op_bra_exc[1:, 0] = 1.0
    op_bra_to_g = np.zeros((nsite_local + 1, nsite_local + 1), dtype=np.float64)
    op_bra_to_g[0, 1:] = 1.0
    if use_sparse_ops:
        op_ket_exc = sparse.coo_matrix(op_ket_exc)
        op_bra_exc = sparse.coo_matrix(op_bra_exc)
        op_bra_to_g = sparse.coo_matrix(op_bra_to_g)

    return (sys_param_local, noise_param_local, hier_param_local, eom_param_local,
            integrator_param_local, psi_k_local, psi_b_local,
            op_ket_exc, op_bra_exc, op_bra_to_g)


def _extract_storage_block(storage_data, t_start, t_end):
    """
    Extracts trajectory storage entries in a strict time window (t_start, t_end].
    """
    t_axis = np.array(storage_data["t_axis"], dtype=float)
    list_block_idx = np.where((t_axis > t_start + 1e-12) &
                              (t_axis <= t_end + 1e-12))[0]
    psi_traj = [np.array(storage_data["psi_traj"][i]) for i in list_block_idx]
    state_list_block = None
    if "state_list" in storage_data:
        state_list_block = [np.array(storage_data["state_list"][i], dtype=int)
                            for i in list_block_idx]
    return t_axis[list_block_idx], psi_traj, state_list_block


def _run_and_compare_checkpoint_flow(tmp_path, use_sparse_ops=False,
                                     adaptive=False, two_checkpoints=False):
    """
    Executes and validates a staged dyadic checkpoint/resume workflow.

    Coverage in this helper includes:
    - dense/sparse operator execution paths,
    - midpoint checkpoint integrity,
    - resumed-block storage equivalence vs uninterrupted run,
    - optional adaptive basis mode,
    - optional two-checkpoint chaining.
    """
    (sys_param_local, noise_param_local, hier_param_local, eom_param_local,
     integrator_param_local, psi_k_local, psi_b_local,
     op_ket_exc, op_bra_exc, op_bra_to_g) = _build_local_dyadic_case(
        use_sparse_ops=use_sparse_ops
    )

    storage_param = {"psi_traj": True, "t_axis": True, "state_list": True}

    traj_ref = DHOPS(
        sys_param_local.copy(),
        noise_param=noise_param_local.copy(),
        hierarchy_param=hier_param_local,
        eom_param=eom_param_local,
        integration_param=integrator_param_local,
        storage_param=storage_param,
    )
    if adaptive:
        traj_ref.make_adaptive(1e-3, 1e-3, list_permanent_sites=[0])

    traj_ref.initialize(psi_k_local, psi_b_local)
    assert len(traj_ref.list_response_norm_sq) == 1
    traj_ref._dyad_operator(op_ket_exc, "ket")
    assert len(traj_ref.list_response_norm_sq) == 2
    traj_ref._dyad_operator(op_bra_exc, "bra")
    assert len(traj_ref.list_response_norm_sq) == 3

    len_before_prop = len(traj_ref.list_response_norm_sq)
    traj_ref.propagate(6.0, 2.0)
    assert len(traj_ref.list_response_norm_sq) == len_before_prop

    traj_ref._dyad_operator(op_bra_to_g, "bra")
    assert len(traj_ref.list_response_norm_sq) == len_before_prop + 1

    len_before_prop = len(traj_ref.list_response_norm_sq)
    traj_ref.propagate(8.0, 2.0)
    assert len(traj_ref.list_response_norm_sq) == len_before_prop

    ckpt1_path = tmp_path / "dyadic_multi_stage_ckpt_1.npz"
    traj_ref.save_checkpoint(str(ckpt1_path))

    # Capture exact checkpoint-point state for pre-resume integrity checks.
    phi_mid = traj_ref.phi.copy()
    t_mid = traj_ref.t
    norm_mid = np.array(traj_ref.list_response_norm_sq, dtype=np.float64)
    t_axis_mid = np.array(traj_ref.storage.data["t_axis"], dtype=float)
    psi_traj_mid = [np.array(psi_step) for psi_step in traj_ref.storage.data["psi_traj"]]
    state_list_mid = [np.array(state, dtype=int)
                      for state in traj_ref.storage.data.get("state_list", [])]

    # Uninterrupted reference continuation.
    len_before_prop = len(traj_ref.list_response_norm_sq)
    traj_ref.propagate(10.0, 2.0)
    assert len(traj_ref.list_response_norm_sq) == len_before_prop
    t_after_first_resume_block = traj_ref.t

    if two_checkpoints:
        ckpt2_path = tmp_path / "dyadic_multi_stage_ckpt_2.npz"
        traj_ref.save_checkpoint(str(ckpt2_path))
        len_before_prop = len(traj_ref.list_response_norm_sq)
        traj_ref.propagate(4.0, 2.0)
        assert len(traj_ref.list_response_norm_sq) == len_before_prop

    phi_expected = traj_ref.phi.copy()
    t_expected = traj_ref.t
    norm_expected = np.array(traj_ref.list_response_norm_sq, dtype=np.float64)
    t_block_ref, psi_block_ref, state_block_ref = _extract_storage_block(
        traj_ref.storage.data, t_mid, t_after_first_resume_block
    )

    # Resume from checkpoint 1 and verify exact restored midpoint state.
    traj_loaded = DHOPS.load_checkpoint(str(ckpt1_path))
    np.testing.assert_allclose(traj_loaded.phi, phi_mid, atol=1e-12)
    assert traj_loaded.t == t_mid
    np.testing.assert_allclose(
        np.array(traj_loaded.list_response_norm_sq, dtype=np.float64),
        norm_mid,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.array(traj_loaded.storage.data["t_axis"], dtype=float),
                               t_axis_mid, atol=1e-12)
    for psi_test, psi_ref in zip(traj_loaded.storage.data["psi_traj"], psi_traj_mid):
        np.testing.assert_allclose(psi_test, psi_ref, atol=1e-12)
    if state_list_mid:
        for state_test, state_ref in zip(traj_loaded.storage.data["state_list"],
                                         state_list_mid):
            np.testing.assert_array_equal(state_test, state_ref)

    len_before_prop = len(traj_loaded.list_response_norm_sq)
    traj_loaded.propagate(10.0, 2.0)
    assert len(traj_loaded.list_response_norm_sq) == len_before_prop

    # Compare resumed storage block against uninterrupted reference block.
    t_block_loaded, psi_block_loaded, state_block_loaded = _extract_storage_block(
        traj_loaded.storage.data, t_mid, t_after_first_resume_block
    )
    np.testing.assert_allclose(t_block_loaded, t_block_ref, atol=1e-12)
    for psi_test, psi_ref in zip(psi_block_loaded, psi_block_ref):
        np.testing.assert_allclose(psi_test, psi_ref, atol=1e-12)
    if state_block_ref is not None and state_block_loaded is not None:
        for state_test, state_ref in zip(state_block_loaded, state_block_ref):
            np.testing.assert_array_equal(state_test, state_ref)

    traj_final = traj_loaded
    if two_checkpoints:
        # Explicitly checkpoint/reload a second time to validate chained resumes.
        ckpt2_loaded_path = tmp_path / "dyadic_multi_stage_ckpt_2_loaded.npz"
        traj_loaded.save_checkpoint(str(ckpt2_loaded_path))
        traj_loaded_2 = DHOPS.load_checkpoint(str(ckpt2_loaded_path))
        np.testing.assert_allclose(traj_loaded_2.phi, traj_loaded.phi, atol=1e-12)
        assert traj_loaded_2.t == traj_loaded.t
        np.testing.assert_allclose(
            np.array(traj_loaded_2.list_response_norm_sq, dtype=np.float64),
            np.array(traj_loaded.list_response_norm_sq, dtype=np.float64),
            atol=1e-12,
        )
        len_before_prop = len(traj_loaded_2.list_response_norm_sq)
        traj_loaded_2.propagate(4.0, 2.0)
        assert len(traj_loaded_2.list_response_norm_sq) == len_before_prop
        traj_final = traj_loaded_2

    np.testing.assert_allclose(traj_final.phi, phi_expected, atol=1e-12)
    assert traj_final.t == t_expected
    np.testing.assert_allclose(
        np.array(traj_final.list_response_norm_sq, dtype=np.float64),
        norm_expected,
        atol=1e-12,
    )


@pytest.mark.parametrize("use_sparse_ops", [False, True])
def test_dyadic_checkpoint_resume_after_multi_stage_ops(tmp_path, use_sparse_ops):
    """
    Validates dense and sparse operator checkpoint/resume equivalence.
    """
    _run_and_compare_checkpoint_flow(
        tmp_path,
        use_sparse_ops=use_sparse_ops,
        adaptive=False,
        two_checkpoints=False,
    )


def test_dyadic_checkpoint_resume_after_multi_stage_ops_adaptive_two_checkpoints(tmp_path):
    """
    Validates adaptive dyadic checkpoint/resume with two consecutive checkpoints.
    """
    _run_and_compare_checkpoint_flow(
        tmp_path,
        use_sparse_ops=False,
        adaptive=True,
        two_checkpoints=True,
    )


def test_dyadic_checkpoint_load_fails_without_storage_dyadic_data(tmp_path):
    """
    Loading a DyadicTrajectory checkpoint must fail if dyadic storage data is missing.
    """
    (sys_param_local, noise_param_local, hier_param_local, eom_param_local,
     integrator_param_local, psi_k_local, psi_b_local,
     op_ket_exc, _, _) = _build_local_dyadic_case(use_sparse_ops=False)

    traj = DHOPS(
        sys_param_local.copy(),
        noise_param=noise_param_local.copy(),
        hierarchy_param=hier_param_local,
        eom_param=eom_param_local,
        integration_param=integrator_param_local,
    )
    traj.initialize(psi_k_local, psi_b_local)
    traj._dyad_operator(op_ket_exc, "ket")
    traj.propagate(4.0, 2.0)

    ckpt_path = tmp_path / "dyadic_missing_storage_dyadic_data_src.npz"
    broken_path = tmp_path / "dyadic_missing_storage_dyadic_data_broken.npz"
    traj.save_checkpoint(str(ckpt_path))

    data = np.load(ckpt_path, allow_pickle=True)
    checkpoint = {
        key: data[key]
        for key in data.files
        if key not in {"storage_dyadic_data", "allow_pickle"}
    }
    np.savez_compressed(broken_path, **checkpoint)

    with pytest.raises(ValueError, match="missing storage_dyadic_data"):
        DHOPS.load_checkpoint(str(broken_path))
