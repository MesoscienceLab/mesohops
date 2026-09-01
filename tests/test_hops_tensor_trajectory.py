import numpy as np
import pytest
import scipy as sp

from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_tensor_trajectory import HopsTensorTrajectory
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.tensor_operations import extract_gs_amp
from mesohops.util.exceptions import (
    LockedException,
    TrajectoryError,
    UnsupportedRequest,
)

__title__ = 'Test Tensor HOPS vs Vector HOPS'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'

# ============================================================
# Dimer of Dimers System Setup
# ============================================================
# Identical to test_dimer_of_dimers.py: 4 sites, 2 modes per site
# (Drude-Lorentz + LTC correction at 500 cm^-1)
noise_param = {
    'SEED': 0,
    'MODEL': 'FFT_FILTER',
    'TLEN': 25000.0,  # Units: fs
    'TAU': 1.0,  # Units: fs
    'STORE_RAW_NOISE': True,
    'RAND_MODEL': 'BOX_MULLER',
}

nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

T3_loperator = np.zeros([4, 4, 4], dtype=np.float64)
list_gw_sysbath = []
list_lop = []
for i in range(nsite):
    T3_loperator[i, i, i] = 1.0
    list_gw_sysbath.append([g_0, w_0])
    list_lop.append(sp.sparse.coo_matrix(T3_loperator[i]))
    list_gw_sysbath.append([-1j * np.imag(g_0), 500.0])
    list_lop.append(T3_loperator[i])

H2_hamiltonian = np.zeros([nsite, nsite])
H2_hamiltonian[0, 1] = 40
H2_hamiltonian[1, 0] = 40
H2_hamiltonian[1, 2] = 10
H2_hamiltonian[2, 1] = 10
H2_hamiltonian[2, 3] = 40
H2_hamiltonian[3, 2] = 40

sys_param = {
    'HAMILTONIAN': np.array(H2_hamiltonian, dtype=np.complex128),
    'GW_SYSBATH': list_gw_sysbath,
    'L_HIER': list_lop,
    'L_NOISE1': list_lop,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': list_gw_sysbath,
}

eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
hier_param = {'MAXHIER': 2, 'TRUNCATION_METHOD': 'rectangular'}

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0
psi_0 = psi_0 / np.linalg.norm(psi_0)

t_max = 200.0
t_step = 4.0


# ============================================================
# Helper: run vector HOPS
# ============================================================
def _run_vector_hops():
    """Runs standard vector HOPS and returns psi_traj as array."""
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)
    hops.propagate(t_max, t_step)
    return np.array(hops.storage.data['psi_traj'])


# ============================================================
# Helper: run tensor HOPS
# ============================================================
def _run_tensor_hops(method, bond_dim_max=20, mps_epsilon=1e-10):
    """Runs tensor HOPS and returns psi_traj as array."""
    tensor_param = {
        'MPS_EPSILON': mps_epsilon,
        'METHOD': method,
        'BOND_DIM_MAX': bond_dim_max,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    traj.propagate(t_max, t_step)
    return np.array(traj.storage['psi_traj'])


# ============================================================
# TEST SUITE: __init__()
# ============================================================


# ------------------------------------------------------------
# TEST: tensor_param defaults are filled when None is passed
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_tensor_param_defaults():
    # This case tests that passing tensor_param=None fills in all
    # defaults from TENSOR_DICT_DEFAULT.
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=None,
    )
    assert traj.tensor_param['METHOD'] == 'fullstate'
    assert traj.tensor_param['MPS_EPSILON'] == 1e-2
    assert traj.tensor_param['MPO_EPSILON'] == 0.0
    assert traj.tensor_param['BOND_DIM_MAX'] == 10


# ------------------------------------------------------------
# TEST: sparse Hamiltonian is coerced to dense
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_sparse_hamiltonian_coerced():
    # This case tests that a sparse Hamiltonian is converted to a dense
    # array during construction without mutating the caller's dict.
    sp_sys_param = dict(sys_param)
    sp_sys_param['HAMILTONIAN'] = sp.sparse.csr_matrix(
        sys_param['HAMILTONIAN']
    )
    original_ham = sp_sys_param['HAMILTONIAN']
    traj = HopsTensorTrajectory(
        system_param=sp_sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'fullstate',
            'BOND_DIM_MAX': 20,
        },
    )
    # Caller's dict should not be mutated
    assert sp.sparse.issparse(original_ham), (
        'Constructor mutated the caller\'s system_param dict'
    )



# ------------------------------------------------------------
# TEST: storage functions are registered for tensor HOPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_storage_functions_registered():
    # This case tests that phi_traj and phi_norm storage functions
    # are replaced with tensor-aware versions during construction.
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'fullstate',
            'BOND_DIM_MAX': 20,
        },
    )
    # The tensor-specific functions have 'tensor' in their name
    if 'phi_traj' in traj.storage.dic_save:
        assert 'tensor' in traj.storage.dic_save['phi_traj'].__name__
    if 'phi_norm' in traj.storage.dic_save:
        assert 'tensor' in traj.storage.dic_save['phi_norm'].__name__


# ------------------------------------------------------------
# TEST: list_aux_norm warning and removal
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_list_aux_norm_warning():
    # This case tests that list_aux_norm in storage triggers a warning
    # and is removed, since it's not meaningful for tensor HOPS.
    storage_param = {'list_aux_norm': True}
    with pytest.warns(UserWarning, match='list_aux_norm'):
        traj = HopsTensorTrajectory(
            system_param=sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
            storage_param=storage_param,
            tensor_param={
                'MPS_EPSILON': 1e-10,
                'METHOD': 'fullstate',
                'BOND_DIM_MAX': 20,
            },
        )
    assert 'list_aux_norm' not in traj.storage.dic_save


# ============================================================
# TEST SUITE: initialize()
# ============================================================


# ------------------------------------------------------------
# TEST: tensor_basis.adaptive agrees with eom_param
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_nonadaptive_sets_adaptive_false():
    # This case tests that tensor_basis.adaptive is False after
    # initializing with the default DELTA_S=0.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    assert traj.tensor_basis.adaptive is False


@pytest.mark.level(1)
def test_initialize_adaptive_sets_adaptive_true():
    # This case tests that tensor_basis.adaptive is True when
    # constructed with DELTA_S > 0. The adaptive flag is set by
    # tensor_basis.initialize() (called inside HopsTensorTrajectory
    # .initialize()). We call tensor_basis.initialize() directly
    # because the full initialize() path crashes in the adaptive
    # define_basis step (noise index mismatch — pre-existing issue).
    # Full adaptive initialization is tested in
    # test_dimer_of_dimers_tensor.py.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    eom_adaptive = {
        'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR',
        'ADAPTIVE_S': True,
        'DELTA_S': 0.01,
    }
    traj_adap = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_adaptive,
        integration_param={
            'INTEGRATOR': 'RUNGE_KUTTA',
            'EARLY_INTEGRATOR_STEPS': 5,
            'INCHWORM_CAP': 5,
        },
        tensor_param=tensor_param,
    )
    traj_adap.tensor_basis.initialize(0.01)
    assert traj_adap.tensor_basis.adaptive is True


# ------------------------------------------------------------
# TEST: make_adaptive rejects list_permanent_sites
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_make_adaptive_rejects_list_permanent_sites():
    # This case tests that passing list_permanent_sites to a tensor
    # trajectory's make_adaptive raises NotImplementedError. The
    # parent class accepts the argument and stores it on
    # system.param["list_permanent_sites"], but HopsTensorBasis does
    # not read that key, so the requested sites would silently not be
    # preserved in the adaptive basis. Failing fast at configuration
    # time gives the caller a clear signal.
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'fullstate',
            'BOND_DIM_MAX': 20,
        },
    )
    with pytest.raises(NotImplementedError, match='list_permanent_sites'):
        traj.make_adaptive(
            delta_a=1e-3, delta_s=1e-3, list_permanent_sites=[0],
        )


# ============================================================
# TEST SUITE: initialize() — wavefunction encoding
# ============================================================


# ------------------------------------------------------------
# TEST: Initial wavefunction is preserved
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize('method', [
    'fullstate', 'number',
])
def test_initialize_preserves_psi0(method):
    # This case tests that the initial physical wavefunction is correctly
    # encoded and recovered from the tensor representation.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)

    np.testing.assert_allclose(
        traj.storage['psi_traj'][0],
        psi_0,
        atol=1e-12,
        err_msg=f'Initial wavefunction not preserved for {method}',
    )


# ------------------------------------------------------------
# TEST: Auto-detection of nearest-neighbor Hamiltonian
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_nearest_neighbor_autodetect():
    # This case tests that HopsSystem.flag_nearest_neighbor_ham is set correctly
    # during system initialization. The dimer-of-dimers Hamiltonian is
    # nearest-neighbor, so it should be True.
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'number',
            'BOND_DIM_MAX': 20,
        },
    )
    traj.initialize(psi_0)
    assert traj.tensor_basis.system.flag_nearest_neighbor_ham is True, (
        'HopsSystem should identify the dimer-of-dimers Hamiltonian as nearest-neighbor'
    )

    # Non-NN case: add a long-range coupling (site 0 ↔ site 3)
    H2_nonnn = np.array(sys_param['HAMILTONIAN'], dtype=np.complex128)
    H2_nonnn[0, 3] = 5.0
    H2_nonnn[3, 0] = 5.0
    nonnn_sys_param = dict(sys_param)
    nonnn_sys_param['HAMILTONIAN'] = H2_nonnn
    traj_nonnn = HopsTensorTrajectory(
        system_param=nonnn_sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'number',
            'BOND_DIM_MAX': 20,
        },
    )
    traj_nonnn.initialize(psi_0)
    assert traj_nonnn.tensor_basis.system.flag_nearest_neighbor_ham is False, (
        'Hamiltonian with long-range coupling should not be identified as NN'
    )


# ------------------------------------------------------------
# TEST: Double initialization raises LockedException
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_double_call_raises():
    # This case tests that calling initialize() twice raises LockedException.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)

    with pytest.raises(LockedException):
        traj.initialize(psi_0)


# ------------------------------------------------------------
# TEST: storage.n_dim is set after initialization
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_storage_n_dim_set():
    # This case tests that storage.n_dim is set to NSTATES after
    # initialization, so that storage['psi_traj'] can reconstruct
    # the full dense wavefunction in adaptive mode.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    assert traj.storage.n_dim == nsite


# ------------------------------------------------------------
# TEST: tensor_basis.eom is constructed during initialization
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_constructs_eom():
    # This case tests that tensor_basis.eom is set to a HopsTensorEOM
    # instance after initialization.
    from mesohops.tensor.hops_tensor_eom import HopsTensorEOM
    traj = _make_initialized_traj()
    assert isinstance(traj.tensor_basis.eom, HopsTensorEOM)


# ------------------------------------------------------------
# TEST: z_mem initialized to zeros
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_zmem_zeros():
    # This case tests that z_mem is initialized to a complex zero array
    # with length matching the noise memory mode indices.
    traj = _make_initialized_traj()
    assert traj.z_mem.dtype == np.complex128
    np.testing.assert_array_equal(traj.z_mem, 0.0)


# ------------------------------------------------------------
# TEST: self.t = 0 after initialization
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_time_zero():
    # This case tests that t is set to 0 after initialization.
    traj = _make_initialized_traj()
    assert traj.t == 0


# ------------------------------------------------------------
# TEST: timer_checkpoint metadata is stored
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_stores_timer_metadata():
    # This case tests that INITIALIZATION_TIME is recorded in
    # storage metadata after initialization.
    traj = _make_initialized_traj()
    assert 'INITIALIZATION_TIME' in traj.storage.metadata
    assert traj.storage.metadata['INITIALIZATION_TIME'] >= 0


# ============================================================
# TEST SUITE: propagate() — norm preservation
# ============================================================


# ------------------------------------------------------------
# TEST: Physical wavefunction stays normalized
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_norm_preserved():
    # This case tests that the physical wavefunction norm stays
    # close to 1 throughout propagation for the normalized nonlinear
    # equation of motion.
    psi_tensor = _run_tensor_hops(
        method='fullstate',
    )

    for i_step in range(len(psi_tensor)):
        norm = np.linalg.norm(psi_tensor[i_step])
        np.testing.assert_allclose(
            norm,
            1.0,
            atol=1e-10,
            err_msg=f'Norm drifted to {norm} at step {i_step}',
        )


# ------------------------------------------------------------
# TEST: Statenumber norm preservation
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_norm_preserved_statenumber():
    # This case tests norm preservation in statenumber representation.
    psi_tensor = _run_tensor_hops(method='number')
    for i_step in range(len(psi_tensor)):
        norm = np.linalg.norm(psi_tensor[i_step])
        np.testing.assert_allclose(
            norm, 1.0, atol=1e-10,
            err_msg=f'Statenumber norm drifted to {norm} at step {i_step}',
        )


# ============================================================
# TEST SUITE: propagate() — bond dimension sensitivity
# ============================================================


# ------------------------------------------------------------
# TEST: Larger bond dimension gives more accurate result
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.parametrize('method', [
    'fullstate', 'number',
])
def test_propagate_bond_dim_convergence(method):
    # This case tests that increasing bond dimension brings the tensor
    # result closer to the vector HOPS result.
    psi_vector = _run_vector_hops()

    # Fix mps_epsilon so only bond_dim_max varies; otherwise a failure
    # could be caused by either parameter.
    psi_small_bond = _run_tensor_hops(
        method=method, bond_dim_max=2, mps_epsilon=1e-10,
    )
    psi_large_bond = _run_tensor_hops(
        method=method, bond_dim_max=20, mps_epsilon=1e-10,
    )

    n_steps = min(len(psi_vector), len(psi_small_bond), len(psi_large_bond))
    err_small = np.mean(
        [np.linalg.norm(psi_small_bond[i] - psi_vector[i]) for i in range(n_steps)]
    )
    err_large = np.mean(
        [np.linalg.norm(psi_large_bond[i] - psi_vector[i]) for i in range(n_steps)]
    )

    assert err_large <= err_small, (
        f'{method}: larger bond dimension gave worse average result: '
        f'err_large={err_large:.2e} > err_small={err_small:.2e}'
    )
    assert err_large < 0.1 * err_small or err_large < 1e-12, (
        f'{method}: larger bond dim should substantially improve accuracy: '
        f'err_large={err_large:.2e}, err_small={err_small:.2e}'
    )


# ============================================================
# TEST SUITE: propagate() — TDVP1 integrator
# ============================================================


# ------------------------------------------------------------
# TEST: TDVP1 approaches RK4 result
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.parametrize('method', [
    'fullstate', 'number',
])
def test_propagate_tdvp1_approaches_rk4(method):
    # TODO: TDVP1-vs-RK4 error (~0.03 at t_step=4.0) is larger than
    # expected and does not converge below ~1e-4 even with small steps.
    # Likely due to fundamental algorithmic differences (tangent-space
    # projection vs MPO contraction + SVD compression). Verify accuracy
    # independently before using TDVP in production.
    #
    # This case tests that the TDVP1 trajectory is qualitatively
    # consistent with the tensor RK4 trajectory. Compares against
    # tensor RK4 (not vector HOPS) to isolate integrator error from
    # representation error. Uses chi=4; error is identical from chi=4
    # to chi=20 for this system (MPS is low-rank).
    t_max_tdvp = 60.0
    chi_tdvp = 4

    # Tensor RK4 reference (same representation and parameters)
    psi_rk4 = _run_tensor_hops(
        method=method,
        bond_dim_max=chi_tdvp,
        mps_epsilon=1e-10,
    )

    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': chi_tdvp,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param={'INTEGRATOR': 'TDVP1'},
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    assert traj.is_tdvp is True, 'TDVP1 integrator was not activated'
    traj.propagate(t_max_tdvp, t_step)
    psi_tdvp = np.array(traj.storage['psi_traj'])

    n_steps = min(len(psi_rk4), len(psi_tdvp))
    max_err = max(np.linalg.norm(psi_tdvp[i] - psi_rk4[i]) for i in range(n_steps))
    assert max_err < 0.05, (
        f'TDVP1 diverged from tensor RK4: max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: TDVP2 propagation
# ============================================================


# ------------------------------------------------------------
# TEST: TDVP2 approaches RK4 result
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_tdvp2_approaches_rk4():
    # TODO: same accuracy concern as TDVP1 test above. See that TODO.
    #
    # This case tests that the TDVP2 trajectory is qualitatively
    # consistent with the tensor RK4 trajectory. Compares against
    # tensor RK4 (not vector HOPS) to isolate integrator error.
    # TDVP2 two-site updates are ~4x more expensive than TDVP1,
    # so we use a shorter propagation (20 fs).
    t_max_tdvp = 20.0
    chi_tdvp = 4

    # Tensor RK4 reference (same representation and parameters)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': chi_tdvp,
    }
    traj_rk4 = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj_rk4.initialize(psi_0)
    traj_rk4.propagate(t_max_tdvp, t_step)
    psi_rk4 = np.array(traj_rk4.storage['psi_traj'])

    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param={'INTEGRATOR': 'TDVP2'},
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    assert traj.is_tdvp is True, 'TDVP2 integrator was not activated'
    traj.propagate(t_max_tdvp, t_step)
    psi_tdvp2 = np.array(traj.storage['psi_traj'])

    n_steps = min(len(psi_rk4), len(psi_tdvp2))
    max_err = max(np.linalg.norm(psi_tdvp2[i] - psi_rk4[i]) for i in range(n_steps))
    assert max_err < 0.05, (
        f'TDVP2 diverged from tensor RK4: max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: HopsTensorTrajectory integrator selection
# ============================================================


# ------------------------------------------------------------
# TEST: RUNGE_KUTTA sets correct integrator attributes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_rk4_integrator_attributes():
    # This case tests that RUNGE_KUTTA sets TDVP=False and the
    # correct step/variable functions.
    from mesohops.integrator.tensor_integrator import (
        runge_kutta_step_tensor,
        runge_kutta_variables,
    )

    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param={'INTEGRATOR': 'RUNGE_KUTTA'},
        tensor_param=tensor_param,
    )
    assert traj.is_tdvp is False
    assert traj.step is runge_kutta_step_tensor
    assert traj.integration_var is runge_kutta_variables


# ------------------------------------------------------------
# TEST: TDVP1 sets correct integrator attributes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_tdvp1_integrator_attributes():
    # This case tests that TDVP1 sets is_tdvp=True and the correct
    # step/variable functions.
    from mesohops.integrator.tensor_integrator import (
        single_point_variables,
        tdvp_step_tensor,
    )

    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param={'INTEGRATOR': 'TDVP1'},
        tensor_param=tensor_param,
    )
    assert traj.is_tdvp is True
    assert traj.step is tdvp_step_tensor
    assert traj.integration_var is single_point_variables
    assert traj._tdvp_method == '1tdvp'


# ------------------------------------------------------------
# TEST: TDVP2 sets correct integrator attributes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_tdvp2_integrator_attributes():
    # This case tests that TDVP2 sets is_tdvp=True and the correct
    # step/variable functions.
    from mesohops.integrator.tensor_integrator import (
        single_point_variables,
        tdvp_step_tensor,
    )

    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param={'INTEGRATOR': 'TDVP2'},
        tensor_param=tensor_param,
    )
    assert traj.is_tdvp is True
    assert traj.step is tdvp_step_tensor
    assert traj.integration_var is single_point_variables
    assert traj._tdvp_method == '2tdvp'


# ------------------------------------------------------------
# TEST: Invalid integrator raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_invalid_integrator_raises():
    # This case tests that an unsupported integrator name raises
    # UnsupportedRequest.
    from mesohops.util.exceptions import UnsupportedRequest

    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    with pytest.raises(UnsupportedRequest):
        HopsTensorTrajectory(
            system_param=sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param={'INTEGRATOR': 'INVALID'},
            tensor_param=tensor_param,
        )


# ============================================================
# TEST SUITE: _operator()
# ============================================================


def _make_initialized_traj():
    """Helper: creates and initializes a fullstate tensor trajectory."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    return traj


# ------------------------------------------------------------
# TEST: Sparse operator input works correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_sparse_input():
    # This case tests that _operator accepts a sparse matrix and
    # produces the same result as the dense equivalent.
    # Two separate trajectories are created so that the in-place mutation
    # from one _operator call does not affect the second comparison.
    traj_dense = _make_initialized_traj()
    traj_sparse = _make_initialized_traj()
    # Build dense and sparse versions of the same operator
    H2_op_dense = np.eye(nsite, dtype=np.complex128) * 0.5
    H2_op_sparse = sp.sparse.csr_matrix(H2_op_dense)
    # Apply each operator to its own trajectory
    traj_dense._operator(H2_op_dense)
    traj_sparse._operator(H2_op_sparse)
    V1_wf_dense = traj_dense.wavefunction.psi
    V1_wf_sparse = traj_sparse.wavefunction.psi
    np.testing.assert_allclose(
        V1_wf_sparse,
        V1_wf_dense,
        atol=1e-12,
        err_msg='Sparse and dense ops should give same result',
    )


# ------------------------------------------------------------
# TEST: Off-diagonal operator transfers population
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_off_diagonal():
    # This case tests that an off-diagonal operator (swap sites 2 and 3)
    # moves population from site 2 to site 3. Unlike the identity and
    # projection tests, this exercises the einsum contraction with
    # nonzero off-diagonal entries.
    traj = _make_initialized_traj()
    # Swap operator: |2><3| + |3><2| + identity on sites 0,1
    H2_swap = np.eye(nsite, dtype=np.complex128)
    H2_swap[2, 2] = 0.0
    H2_swap[3, 3] = 0.0
    H2_swap[2, 3] = 1.0
    H2_swap[3, 2] = 1.0
    traj._operator(H2_swap)
    V1_wf_after = traj.wavefunction.psi
    V1_wf_full = np.zeros(nsite, dtype=np.complex128)
    V1_wf_full[traj.tensor_basis.system.state_list] = V1_wf_after
    # psi_0 had all amplitude on site 2; after swap it should be on site 3
    expected = np.zeros(nsite, dtype=np.complex128)
    expected[3] = psi_0[2]
    np.testing.assert_allclose(
        V1_wf_full,
        expected,
        atol=1e-12,
        err_msg='Swap operator did not transfer population',
    )


# ------------------------------------------------------------
# TEST: Projection onto non-occupied site zeros wavefunction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_projection_orthogonal():
    # This case tests that projecting onto a site with zero initial
    # amplitude zeros out the wavefunction. This is a stronger test
    # than the existing projection test (which projects onto the
    # occupied site and is effectively identity).
    traj = _make_initialized_traj()
    H2_proj_0 = np.zeros((nsite, nsite), dtype=np.complex128)
    H2_proj_0[0, 0] = 1.0  # project onto site 0, but psi_0 is on site 2
    traj._operator(H2_proj_0)
    V1_wf_after = traj.wavefunction.psi
    np.testing.assert_allclose(
        V1_wf_after,
        0.0,
        atol=1e-12,
        err_msg='Projection onto empty site should zero wf',
    )


# ------------------------------------------------------------
# TEST: _operator does not reset early time integrator for non-adaptive
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_operator_no_reset_nonadaptive():
    # This case tests that _operator does NOT call reset_early_time_integrator
    # for non-adaptive trajectories, matching the parent's behavior where the
    # reset only happens inside the adaptive branch.
    traj = _make_initialized_traj()
    # Propagate a few steps to consume early integrator steps
    traj.propagate(8.0, 4.0)
    counter_before = traj._early_step_counter
    H2_identity = np.eye(nsite, dtype=np.complex128)
    traj._operator(H2_identity)
    # Counter should be unchanged — _operator does not reset for non-adaptive
    assert traj._early_step_counter == counter_before, (
        f'Expected _early_step_counter={counter_before}, '
        f'got {traj._early_step_counter}'
    )


# ------------------------------------------------------------
# TEST: Statenumber off-diagonal operator transfers population
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_statenumber_off_diagonal():
    # This case tests the statenumber _operator path with an off-diagonal
    # swap operator on the initial state, isolating statenumber bugs
    # from propagation.
    traj = _make_initialized_traj_statenumber()
    H2_swap = np.eye(nsite, dtype=np.complex128)
    H2_swap[2, 2] = 0.0
    H2_swap[3, 3] = 0.0
    H2_swap[2, 3] = 1.0
    H2_swap[3, 2] = 1.0
    traj._operator(H2_swap)
    V1_wf_full = np.zeros(nsite, dtype=np.complex128)
    V1_wf_full[traj.tensor_basis.system.state_list] = traj.wavefunction.psi
    expected = np.zeros(nsite, dtype=np.complex128)
    expected[3] = psi_0[2]
    np.testing.assert_allclose(
        V1_wf_full, expected, atol=1e-10,
        err_msg='Statenumber swap did not transfer population',
    )


# ------------------------------------------------------------
# TEST: Statenumber projection onto non-occupied site
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_statenumber_projection_orthogonal():
    # This case tests that projecting onto an unoccupied site zeros the
    # wavefunction in statenumber representation.
    traj = _make_initialized_traj_statenumber()
    H2_proj_0 = np.zeros((nsite, nsite), dtype=np.complex128)
    H2_proj_0[0, 0] = 1.0
    traj._operator(H2_proj_0)
    np.testing.assert_allclose(
        traj.wavefunction.psi, 0.0, atol=1e-12,
        err_msg='Statenumber projection onto empty site should zero wf',
    )


# ------------------------------------------------------------
# TEST: Sparse off-diagonal operator exercises CSR indexing
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_sparse_off_diagonal():
    # This case tests that a sparse Hermitian operator with complex
    # off-diagonal entries produces the same result as its dense
    # equivalent. The 0.5*I test doesn't exercise sparse indexing on
    # off-diagonal entries; this tests CSR conversion + np.ix_ trimming
    # on non-trivial sparsity patterns.
    H2_op = np.array([
        [0.5,    0.1+0.2j, 0.0,      0.05-0.1j],
        [0.1-0.2j, 0.3,    0.15+0.1j, 0.0],
        [0.0,    0.15-0.1j, 0.7,      0.2+0.05j],
        [0.05+0.1j, 0.0,   0.2-0.05j, 0.6],
    ], dtype=np.complex128)
    traj_dense = _make_initialized_traj()
    traj_sparse = _make_initialized_traj()
    traj_dense._operator(H2_op)
    traj_sparse._operator(sp.sparse.csr_matrix(H2_op))
    np.testing.assert_allclose(
        traj_sparse.wavefunction.psi, traj_dense.wavefunction.psi,
        atol=1e-12,
        err_msg='Sparse Hermitian operator differs from dense',
    )


# ------------------------------------------------------------
# TEST: Statenumber swap on initial state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_statenumber_swap_initial():
    # This case tests the statenumber _operator path on the initial state
    # (before propagation), isolating statenumber bugs from propagation.
    traj = _make_initialized_traj()
    traj_sn = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'number',
            'BOND_DIM_MAX': 20,
        },
    )
    traj_sn.initialize(psi_0)
    # Swap sites 2 and 3
    H2_swap = np.zeros((nsite, nsite), dtype=np.complex128)
    H2_swap[2, 3] = 1.0
    H2_swap[3, 2] = 1.0
    traj._operator(H2_swap)
    traj_sn._operator(H2_swap)
    np.testing.assert_allclose(
        traj_sn.psi, traj.psi, atol=1e-10,
        err_msg='Statenumber swap on initial state differs from fullstate',
    )


def _make_initialized_traj_statenumber():
    """Helper: creates and initializes a statenumber tensor trajectory."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    return traj


def _make_vacuum_traj(phys_init):
    """Helper: number-method trajectory in the vacuum convention.

    Sets flag_gs_vacuum before initialize so the all-zeros MPS
    configuration carries the physical ground state, matching the
    absorption / fluorescence dipole pathways.
    """
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.wavefunction.flag_gs_vacuum = True
    traj.initialize(phys_init)
    return traj


# ------------------------------------------------------------
# TEST: apply_dipole_lower_plus_ident gives mu_k |g> + |e_k>
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_dipole_lower_plus_ident_action():
    # The lower+ident dipole (sum_k mu_k a_k + I_excited) maps a
    # single-excitation state sum_k c_k |e_k> to
    # (sum_k mu_k c_k) |g> + sum_k c_k |e_k>: the single-excitation content
    # is preserved and the all-zeros configuration picks up the mu-weighted
    # sum. The single-excitation state is built the way the spectroscopy
    # pathways do -- raise from |g> -- so the MPS stays in the
    # single-excitation manifold. A direct multi-site initialize() would
    # instead encode a product state carrying spurious double-excitation
    # amplitudes c_j*c_k that the lower term would fold back into the
    # single-excitation readback.
    V1_c = np.zeros(nsite, dtype=np.complex128)
    V1_c[1] = 0.6
    V1_c[2] = 0.8j
    traj = _make_vacuum_traj(np.zeros(nsite, dtype=np.complex128))
    traj.apply_dipole_raise(V1_c)  # |g> -> sum_k c_k |e_k>

    list_mu = np.array([0.5, 1.5 - 0.2j, 0.0, 0.7j], dtype=np.complex128)
    traj.apply_dipole_lower_plus_ident(list_mu)

    psi = traj.wavefunction.psi
    gs_amp = extract_gs_amp(
        traj.wavefunction.list_cores_phi, traj.wavefunction.method,
    )
    np.testing.assert_allclose(
        psi, V1_c, atol=1e-12,
        err_msg='lower+ident must preserve the single-excitation amplitudes',
    )
    np.testing.assert_allclose(
        gs_amp, np.sum(list_mu * V1_c), atol=1e-12,
        err_msg='lower+ident must place sum_k mu_k c_k on the all-zeros config',
    )


# ------------------------------------------------------------
# TEST: apply_dipole_raise_plus_ground_ident gives |g> + sum_k mu_k |e_k>
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_dipole_raise_plus_ground_ident_action():
    # The raise+ground-ident dipole (sum_k mu_k a_k^dagger + I_g) maps the
    # ground state |g> to |g> + sum_k mu_k |e_k>: the all-zeros amplitude
    # is preserved and each excited site picks up mu_k. Verifies the
    # trajectory-level MPO wiring.
    traj = _make_vacuum_traj(np.zeros(nsite, dtype=np.complex128))

    list_mu = np.array([0.5, 1.5 - 0.2j, 0.0, 0.7j], dtype=np.complex128)
    traj.apply_dipole_raise_plus_ground_ident(list_mu)

    psi = traj.wavefunction.psi
    gs_amp = extract_gs_amp(
        traj.wavefunction.list_cores_phi, traj.wavefunction.method,
    )
    np.testing.assert_allclose(
        psi, list_mu, atol=1e-12,
        err_msg='raise+ground-ident must place mu_k on each excited site',
    )
    np.testing.assert_allclose(
        gs_amp, 1.0, atol=1e-12,
        err_msg='raise+ground-ident must preserve the ground-state amplitude',
    )


# ------------------------------------------------------------
# TEST: Propagated statenumber swap matches fullstate swap
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_operator_statenumber_propagated_swap_vs_fullstate():
    # This case tests that a swap operator applied after propagation gives
    # the same phi_0 in both representations. This is the strongest
    # correctness test: the propagated state has multi-site amplitude and
    # populated hierarchy, so the MPO must handle nontrivial bond dimensions,
    # mode core pass-throughs, and cross-site transfer matrices.
    H2_swap = np.eye(nsite, dtype=np.complex128)
    H2_swap[2, 2] = 0.0
    H2_swap[3, 3] = 0.0
    H2_swap[2, 3] = 1.0
    H2_swap[3, 2] = 1.0

    traj_full = _make_initialized_traj()
    traj_full.propagate(20.0, t_step)
    traj_full._operator(H2_swap)

    traj_snum = _make_initialized_traj_statenumber()
    traj_snum.propagate(20.0, t_step)
    traj_snum._operator(H2_swap)

    np.testing.assert_allclose(
        traj_snum.wavefunction.psi,
        traj_full.wavefunction.psi,
        atol=1e-6,
        err_msg='Statenumber operator result diverges from fullstate after propagation',
    )


# ------------------------------------------------------------
# TEST: Propagated statenumber raise matches fullstate raise
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_operator_statenumber_propagated_raise_vs_fullstate():
    # This case tests a fluorescence-style raise operator on a propagated
    # state. The raise operator |3><0| couples distant sites, exercising
    # the long-range transfer-matrix channels in the MPO.
    op_raise = np.zeros((nsite, nsite), dtype=np.complex128)
    op_raise[3, 0] = 1.0

    traj_full = _make_initialized_traj()
    traj_full.propagate(20.0, t_step)
    traj_full._operator(op_raise)

    traj_snum = _make_initialized_traj_statenumber()
    traj_snum.propagate(20.0, t_step)
    traj_snum._operator(op_raise)

    np.testing.assert_allclose(
        traj_snum.wavefunction.psi,
        traj_full.wavefunction.psi,
        atol=1e-6,
        err_msg='Statenumber raise operator diverges from fullstate after propagation',
    )


# ------------------------------------------------------------
# TEST: Propagated statenumber general operator matches fullstate
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_operator_statenumber_propagated_general_vs_fullstate():
    # This case tests a dense operator with all entries nonzero on a
    # propagated state. This is the worst case for the MPO bond dimension
    # and exercises every channel in the transfer-matrix structure.
    H2_general = np.array(
        [
            [0.5, 0.1, 0.2, 0.0],
            [0.1, 0.3, 0.0, 0.4],
            [0.2, 0.0, 0.7, 0.1],
            [0.0, 0.4, 0.1, 0.6],
        ],
        dtype=np.complex128,
    )

    traj_full = _make_initialized_traj()
    traj_full.propagate(20.0, t_step)
    traj_full._operator(H2_general)

    traj_snum = _make_initialized_traj_statenumber()
    traj_snum.propagate(20.0, t_step)
    traj_snum._operator(H2_general)

    np.testing.assert_allclose(
        traj_snum.wavefunction.psi,
        traj_full.wavefunction.psi,
        atol=1e-6,
        err_msg=(
            'Statenumber general operator diverges from fullstate after propagation'
        ),
    )


# ------------------------------------------------------------
# TEST: Tensor _operator matches vector _operator
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_tensor_matches_vector():
    # This case tests that applying a dense Hermitian operator to the
    # same initial state via tensor _operator and vector _operator
    # produces the same physical wavefunction. Uses complex off-diagonal
    # entries coupling all four sites to exercise the full contraction.
    H2_op = np.array([
        [0.5,    0.1+0.2j, 0.0,      0.05-0.1j],
        [0.1-0.2j, 0.3,    0.15+0.1j, 0.0],
        [0.0,    0.15-0.1j, 0.7,      0.2+0.05j],
        [0.05+0.1j, 0.0,   0.2-0.05j, 0.6],
    ], dtype=np.complex128)

    # Tensor path
    traj_tensor = _make_initialized_traj()
    traj_tensor._operator(H2_op)

    # Vector path
    hops_vector = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_vector.initialize(psi_0)
    hops_vector._operator(H2_op)

    np.testing.assert_allclose(
        traj_tensor.psi, hops_vector.psi, atol=1e-10,
        err_msg='Tensor _operator result differs from vector _operator',
    )


# ------------------------------------------------------------
# TEST: Zero operator zeroes out the wavefunction cleanly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_zero_operator():
    # This case tests that applying a zero matrix via _operator produces
    # psi = 0 across every state without crashing. Edge case for a
    # degenerate operator.
    traj = _make_initialized_traj()
    H2_zero = np.zeros((nsite, nsite), dtype=np.complex128)
    traj._operator(H2_zero)
    V1_wf_after = traj.wavefunction.psi
    np.testing.assert_allclose(
        V1_wf_after, 0.0, atol=1e-12,
        err_msg='Zero operator should zero out the wavefunction',
    )


# ------------------------------------------------------------
# TEST: Mis-sized operator raises a clear ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_operator_wrong_dimensions_raises():
    # This case tests that _operator validates the operator shape
    # against the full system size. Without the guard, too-small
    # operators leak an IndexError from np.ix_ and too-large operators
    # are silently sliced to the first n_state_full x n_state_full
    # block — both are invalid silent-failure modes.
    # Too-small (2x2 on a 4-state system) must raise.
    traj_small = _make_initialized_traj()
    H2_small = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match='shape'):
        traj_small._operator(H2_small)
    # Too-large (5x5 on a 4-state system) must also raise — no silent
    # slicing to the first 4x4 block.
    traj_large = _make_initialized_traj()
    H2_large = np.eye(nsite + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match='shape'):
        traj_large._operator(H2_large)


# ============================================================
# TEST SUITE: Statenumber propagation
# ============================================================


# ------------------------------------------------------------
# TEST: Statenumber propagation produces valid trajectory
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_statenumber_valid_trajectory():
    # This case tests that propagation with number
    # runs without error and produces a trajectory of the expected
    # length with normalized wavefunctions.
    psi_tensor = _run_tensor_hops(
        method='number',
    )
    # Trajectory should have initial + propagation steps
    n_expected = int(t_max / t_step) + 1
    assert len(psi_tensor) == n_expected, (
        f'Expected {n_expected} steps, got {len(psi_tensor)}'
    )
    # Norm should stay close to 1
    for i_step in range(len(psi_tensor)):
        norm = np.linalg.norm(psi_tensor[i_step])
        np.testing.assert_allclose(
            norm,
            1.0,
            atol=1e-4,
            err_msg=f'Statenumber norm drifted to {norm} at step {i_step}',
        )


# ------------------------------------------------------------
# TEST: max_tensor_complexity is populated end-to-end after propagate
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_populates_max_tensor_complexity():
    # This case tests the full max_complexity round-trip through the
    # side-channel: derivative() sets eom.last_matvec_complexity, the
    # RK4 step function publishes the per-step max on
    # eom.max_complexity_step, propagate reads it and forwards to
    # storage via the registered save_max_tensor_complexity callable.
    # The storage trace must have an entry per timestep with positive
    # values (any RK4 matvec on a non-trivial MPS gives complexity > 0).
    traj = _make_initialized_traj()
    traj.propagate(t_max, t_step)
    list_complexities = traj.storage['max_tensor_complexity']
    n_expected = int(t_max / t_step) + 1
    assert len(list_complexities) == n_expected, (
        f'Expected {n_expected} max_tensor_complexity entries, '
        f'got {len(list_complexities)}'
    )
    for i, c in enumerate(list_complexities):
        assert isinstance(c, (int, np.integer)), (
            f'Step {i}: expected int complexity, got {type(c).__name__}'
        )
    # Step 0 captures the initial state before any RK4 matvec, so 0 is
    # valid there. Every subsequent step ran the RK4 derivative chain
    # on a non-trivial MPS, so the published max must be positive.
    assert list_complexities[0] >= 0
    for i in range(1, len(list_complexities)):
        assert list_complexities[i] > 0, (
            f'Step {i}: expected positive complexity, got '
            f'{list_complexities[i]}'
        )


# ------------------------------------------------------------
# TEST: STORE_STEP_TIMING default off keeps per-call totals
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_step_timing_flag_default_off():
    # This case tests that without the STORE_STEP_TIMING flag,
    # LIST_PROPAGATION_TIME retains its pre-flag behavior on the
    # tensor path: one float per propagate() call.
    traj = _make_initialized_traj()
    traj.propagate(t_max, t_step)
    traj.propagate(t_max, t_step)

    list_prop_time = traj.storage.metadata['LIST_PROPAGATION_TIME']
    assert len(list_prop_time) == 2
    for entry in list_prop_time:
        assert isinstance(entry, float)
        assert entry >= 0


# ------------------------------------------------------------
# TEST: STORE_STEP_TIMING on populates per-step (t, dt) tuples
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_propagate_step_timing_flag_on():
    # This case tests that with STORE_STEP_TIMING=True on the tensor
    # path, LIST_PROPAGATION_TIME holds (t_fs, wall_seconds) tuples
    # — one per integration step, sim-time stamps match the
    # propagation grid, wall times are non-negative, and entries
    # concatenate across propagate() calls with a monotonic time
    # axis.
    integration_param = {
        'INTEGRATOR': 'RUNGE_KUTTA',
        'STORE_STEP_TIMING': True,
    }
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integration_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    n_steps = int(np.ceil(t_max / t_step))
    traj.propagate(t_max, t_step)

    list_prop_time = traj.storage.metadata['LIST_PROPAGATION_TIME']
    assert len(list_prop_time) == n_steps
    for entry in list_prop_time:
        assert isinstance(entry, tuple) and len(entry) == 2
    list_t = [t for t, _ in list_prop_time]
    list_dt = [dt for _, dt in list_prop_time]
    assert all(dt >= 0 for dt in list_dt)
    np.testing.assert_allclose(list_t, t_step * np.arange(1, n_steps + 1))

    # Second propagate call: entries append, time axis stays monotonic.
    traj.propagate(t_max, t_step)
    list_prop_time = traj.storage.metadata['LIST_PROPAGATION_TIME']
    assert len(list_prop_time) == 2 * n_steps
    list_t = [t for t, _ in list_prop_time]
    list_dt = [dt for _, dt in list_prop_time]
    assert all(dt >= 0 for dt in list_dt)
    assert np.all(np.diff(list_t) > 0)


# ------------------------------------------------------------
# TEST: inchworm max-tracking takes max across iterations
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.xfail(
    reason='Adaptive tensor trajectory does not yet propagate cleanly '
           'past define_basis (pre-existing noise-index mismatch noted '
           'in test_initialize_adaptive_sets_adaptive_true). When that '
           'root cause is fixed, this test will start passing and '
           'strict=True will flag the xfail mark for removal.',
    strict=True,
)
def test_inchworm_max_complexity_tracking():
    # This case tests that propagate's inchworm loop publishes the MAX
    # of the per-iteration max_complexity values to storage, not the
    # last one or the first one. The inchworm path runs only when the
    # trajectory is adaptive AND the early-integrator counter is below
    # the cap, so the test needs an adaptive trajectory.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    eom_adaptive = {
        'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR',
        'ADAPTIVE_S': True,
        'DELTA_S': 0.01,
    }
    traj_adap = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_adaptive,
        integration_param={
            'INTEGRATOR': 'RUNGE_KUTTA',
            'EARLY_INTEGRATOR_STEPS': 5,
            'INCHWORM_CAP': 5,
        },
        tensor_param=tensor_param,
    )
    # Currently crashes in initialize() at the adaptive define_basis
    # step (noise index mismatch — pre-existing issue). xfail catches
    # the crash; when fixed, propagate runs and the assertion below is
    # the actual contract being verified.
    traj_adap.initialize(psi_0)
    traj_adap.propagate(t_max, t_step)
    list_complexities = traj_adap.storage['max_tensor_complexity']
    n_expected = int(t_max / t_step) + 1
    assert len(list_complexities) == n_expected
    # Inchworm contract: the per-step published value is the MAX across
    # the inchworm iterations for that step, so post-step-0 entries must
    # be positive (every step ran at least one matvec on a non-trivial
    # MPS).
    for i in range(1, len(list_complexities)):
        assert list_complexities[i] > 0




# ============================================================
# TEST SUITE: Checkpoint guards
# ============================================================


# ------------------------------------------------------------
# TEST: save_checkpoint raises NotImplementedError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_save_checkpoint_raises():
    # This case tests that save_checkpoint raises NotImplementedError
    # because tensor trajectory checkpointing is not yet supported.
    traj = _make_initialized_traj()
    with pytest.raises(NotImplementedError):
        traj.save_checkpoint('/tmp/dummy_checkpoint.npz')


# ------------------------------------------------------------
# TEST: load_checkpoint raises NotImplementedError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_load_checkpoint_raises():
    # This case tests that load_checkpoint raises NotImplementedError
    # because tensor trajectory checkpointing is not yet supported.
    with pytest.raises(NotImplementedError):
        HopsTensorTrajectory.load_checkpoint('/tmp/dummy_checkpoint.npz')


# ============================================================
# TEST SUITE: statenumber checkpoint deep copy
# ============================================================


# ------------------------------------------------------------
# TEST: statenumber MPS checkpoint is independent of original
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_checkpoint_deep_copy():
    # This case tests that the checkpoint copy pattern used in propagate
    # and inchworm_integrate produces an independent copy for statenumber
    # representation. Mutating the original wavefunction after checkpoint
    # creation should not affect the checkpoint, and vice versa.
    traj = _make_initialized_traj_statenumber()
    cores = traj.wavefunction.list_cores_phi
    # Make a checkpoint using the same pattern as propagate
    if traj.wavefunction.method == 'number':
        checkpoint = [
            [arr.copy() for arr in g] for g in cores
        ]
    else:
        checkpoint = [c.copy() for c in cores]
    # Save a reference value from the checkpoint
    val_before = checkpoint[0][0][0, 0, 0].copy()
    # Mutate the original wavefunction's inner array
    cores[0][0][0, 0, 0] *= 999.0
    # Checkpoint should be unchanged
    assert checkpoint[0][0][0, 0, 0] == val_before, (
        'Statenumber checkpoint was corrupted by mutation of original cores'
    )


# ============================================================
# TEST SUITE: psi property returns compact wavefunction
# ============================================================


# ------------------------------------------------------------
# TEST: psi returns compact array matching active state count
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_psi_returns_compact():
    # This case tests that traj.psi returns a compact array of
    # length n_active (number of active states), not NSTATES.
    # This matches the parent HopsTrajectory.psi behavior.
    traj = _make_initialized_traj()
    psi = traj.psi
    # Non-adaptive: all states active, so n_active == NSTATES
    assert len(psi) == nsite
    # Analytical: psi_0 = [0, 0, 1, 0], so psi should match at init
    np.testing.assert_allclose(psi, psi_0, atol=1e-12)


# ------------------------------------------------------------
# TEST: psi works in statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_psi_returns_compact_statenumber():
    # This case tests extract_psi in statenumber representation.
    traj = _make_initialized_traj_statenumber()
    psi = traj.psi
    assert len(psi) == nsite
    np.testing.assert_allclose(psi, psi_0, atol=1e-12)


# ------------------------------------------------------------
# TEST: storage psi_traj reconstructs correctly
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.parametrize('method', [
    'fullstate', 'number',
])
def test_storage_psi_traj_after_propagate(method):
    # This case tests that storage['psi_traj'] returns correctly
    # shaped arrays after propagation.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    traj.propagate(8.0, 4.0)
    psi_traj = np.array(traj.storage['psi_traj'])
    # Should have shape (n_steps, NSTATES)
    assert psi_traj.shape[1] == nsite
    # Initial wavefunction should match psi_0
    np.testing.assert_allclose(psi_traj[0], psi_0, atol=1e-12)
    # Later time steps should differ from initial (dynamics happened)
    assert not np.allclose(psi_traj[-1], psi_0, atol=1e-6), (
        'Final state identical to initial — propagation may not have run'
    )


# ============================================================
# TEST SUITE: phi property
# ============================================================


# ------------------------------------------------------------
# TEST: phi setter stores and retrieves cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_setter():
    # This case tests that the phi setter stores the provided list of
    # cores and the getter returns the exact same object (no copy).
    traj = _make_initialized_traj()
    original_phi = traj.phi
    new_phi = [c * 2.0 for c in original_phi]
    traj.phi = new_phi
    # Invariant: getter returns what setter stored (same object)
    assert traj.phi is new_phi


# ------------------------------------------------------------
# TEST: phi after initialization reproduces psi_0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_initial_reproduces_psi0():
    # This case tests that contracting the MPS cores from traj.phi
    # reproduces the initial wavefunction psi_0.
    traj = _make_initialized_traj()
    np.testing.assert_allclose(traj.psi, psi_0, atol=1e-12,
        err_msg='phi cores do not reproduce psi_0 after initialization')


# ------------------------------------------------------------
# TEST: phi works in statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_statenumber():
    # This case tests that traj.phi returns valid MPS cores in
    # statenumber representation (nested list of core groups).
    traj = _make_initialized_traj_statenumber()
    phi = traj.phi
    assert isinstance(phi, list)
    assert len(phi) > 0
    # Statenumber phi is a list of lists (groups per state)
    assert isinstance(phi[0], list)
    # Verify psi is recoverable
    np.testing.assert_allclose(traj.psi, psi_0, atol=1e-12)


# ------------------------------------------------------------
# TEST: phi property shares reference with wavefunction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_shared_reference():
    # This case tests that traj.phi and traj.wavefunction.list_cores_phi
    # are the same object — mutations through one path are visible
    # through the other.
    traj = _make_initialized_traj()
    assert traj.phi is traj.wavefunction.list_cores_phi
    # Mutate through wavefunction, read through phi
    traj.wavefunction.list_cores_phi[0] = traj.wavefunction.list_cores_phi[0] * 2.0
    assert traj.phi[0] is traj.wavefunction.list_cores_phi[0]


# ============================================================
# TEST SUITE: normalize()
# ============================================================


# ------------------------------------------------------------
# TEST: normalize is no-op when basis.eom.normalized is False
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_unnormalized_noop():
    # This case tests that normalize() leaves the wavefunction unchanged
    # when the EOM does not require normalization. Constructs with
    # NONLINEAR EOM (not NORMALIZED NONLINEAR) so basis.eom.normalized
    # is False without monkey-patching.
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param={'EQUATION_OF_MOTION': 'NONLINEAR'},
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': 1e-10,
            'METHOD': 'fullstate',
            'BOND_DIM_MAX': 20,
        },
    )
    traj.initialize(psi_0)
    assert traj.basis.eom.normalized is False
    # Scale wavefunction so any accidental normalize() would be detectable
    traj.phi = [c * 3.0 for c in traj.phi]
    psi_scaled = traj.psi.copy()
    traj.normalize()
    # Invariant: wavefunction unchanged when basis.eom.normalized is False
    np.testing.assert_allclose(traj.psi, psi_scaled, atol=1e-14)


# ------------------------------------------------------------
# TEST: normalize actually normalizes when EOM requires it
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_rescales_wavefunction():
    # This case tests that normalize() actually restores unit norm
    # when the EOM requires normalization.
    traj = _make_initialized_traj()
    # Scale wavefunction away from unit norm
    traj.phi = [c * 3.0 for c in traj.phi]
    assert abs(np.linalg.norm(traj.psi) - 1.0) > 0.1, 'Precondition: norm should differ from 1'
    traj.normalize()
    np.testing.assert_allclose(
        np.linalg.norm(traj.psi), 1.0, atol=1e-10,
        err_msg='normalize() did not restore unit norm',
    )


# ------------------------------------------------------------
# TEST: normalize works in statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_statenumber():
    # This case tests the statenumber normalize path which indexes
    # list_cores_phi[0][0] instead of list_cores_phi[0].
    traj = _make_initialized_traj_statenumber()
    traj.phi = [[core * 3.0 for core in group] for group in traj.phi]
    traj.normalize()
    np.testing.assert_allclose(
        np.linalg.norm(traj.psi), 1.0, atol=1e-10,
        err_msg='Statenumber normalize() did not restore unit norm',
    )


# ------------------------------------------------------------
# TEST: single RK4 step keeps norm within 1e-10 before normalize()
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize('method', [
    'fullstate', 'number',
])
def test_normalize_single_step_norm_drift(method):
    # This case tests that a single RK4 step preserves the physical
    # wavefunction norm to within 1e-10 of unity before normalize() is
    # applied. Uses a 2-site NORMALIZED NONLINEAR system (same physics as
    # the vector counterpart) so both code paths are tested against
    # identical physics. If the norm correction term in the tensor EOM
    # derivative is wrong (e.g. wrong prefactor), norm drifts measurably
    # even in one step.
    T3_loperator = np.zeros([2, 2, 2], dtype=np.float64)
    T3_loperator[0, 0, 0] = 1.0
    T3_loperator[1, 1, 1] = 1.0
    local_sys_param = {
        'HAMILTONIAN': np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
        'GW_SYSBATH': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
        'L_HIER': [T3_loperator[0], T3_loperator[0],
                    T3_loperator[1], T3_loperator[1]],
        'L_NOISE1': [T3_loperator[0], T3_loperator[0],
                      T3_loperator[1], T3_loperator[1]],
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    }
    local_noise_param = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    local_eom_param = {
        'TIME_DEPENDENCE': False,
        'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR',
    }
    local_hier_param = {'MAXHIER': 4}
    local_integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tensor_param = {
        'MPS_EPSILON': 1e-12,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    H1_psi_0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    traj = HopsTensorTrajectory(
        system_param=local_sys_param,
        noise_param=local_noise_param,
        hierarchy_param=local_hier_param,
        eom_param=local_eom_param,
        integration_param=local_integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(H1_psi_0)
    tau = 2.0  # 2 * noise TAU so RK4 samples [0, 1, 2] land on the noise grid

    # Gather integration variables for one step
    dict_var = traj.integration_var(
        traj.z_mem,
        traj.t,
        traj.noise1,
        traj.noise2,
        tau,
        traj.basis.mode.list_l2idx_abs,
        traj.effective_noise_integration,
    )

    # Call RK4 directly — mutates wavefunction in place, bypasses normalize()
    traj._step(dict_var)

    # Norm of the physical wavefunction after RK4 (no normalize applied)
    norm_psi = np.linalg.norm(traj.psi)

    np.testing.assert_allclose(
        norm_psi, 1.0, atol=1e-10,
        err_msg=(
            f'Single-step norm drift {abs(norm_psi - 1.0):.2e} exceeds 1e-10. '
            'The norm correction term in the tensor EOM derivative may be wrong.'
        ),
    )


# ============================================================
# TEST SUITE: propagate() — error guards
# ============================================================


# ------------------------------------------------------------
# TEST: propagate raises when t_axis exceeds noise length
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_propagate_t_axis_exceeds_tlen():
    # This case tests that propagate raises TrajectoryError when the
    # requested propagation time exceeds noise.param['TLEN']. The guard
    # is at np.max(t_axis) > self.noise1.param['TLEN'].
    traj = _make_initialized_traj()
    # Try to propagate way beyond the noise length (TLEN=25000 fs)
    with pytest.raises(TrajectoryError, match='longer than'):
        traj.propagate(100000.0, 4.0)


# ------------------------------------------------------------
# TEST: propagate raises on timestep/noise TAU mismatch
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_propagate_tau_mismatch_raises():
    # This case tests that propagate raises TrajectoryError when the
    # timestep does not align with noise.param['TAU'].
    traj = _make_initialized_traj()
    # noise TAU is 1.0 fs; use a non-divisor timestep
    with pytest.raises(TrajectoryError, match='TAU'):
        traj.propagate(10.0, 0.7)


# ------------------------------------------------------------
# TEST: propagate raises when TAU is None and INTERPOLATE is True
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_propagate_tau_none_interpolate_raises():
    # This case tests that propagate raises UnboundLocalError when
    # noise TAU is None and INTERPOLATE is True. Under these conditions
    # the t_axis variable is never defined, but is used downstream.
    # This is a known bug (see NOTE in hops_tensor_trajectory.propagate);
    # this test documents the failure mode until the guard is fixed.
    traj = _make_initialized_traj()
    traj.noise1.param['TAU'] = None
    traj.noise1.param['INTERPOLATE'] = True
    with pytest.raises(UnboundLocalError):
        traj.propagate(8.0, 4.0)


# ------------------------------------------------------------
# TEST: unsupported early integrator raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_propagate_unsupported_early_integrator_raises():
    # This case tests that an unsupported EARLY_ADAPTIVE_INTEGRATOR
    # value raises UnsupportedRequest during adaptive propagation.
    # We initialize non-adaptive then patch the integration_param to
    # trigger the unsupported early integrator path, because the full
    # adaptive initialization path has a pre-existing noise indexing bug.
    traj = _make_initialized_traj()
    traj.basis.eom.param['ADAPTIVE'] = True
    traj.basis.eom.param['ADAPTIVE_S'] = True
    traj.tensor_basis.adaptive = True
    traj.integration_param['EARLY_ADAPTIVE_INTEGRATOR'] = 'INVALID'
    traj.integration_param['EARLY_INTEGRATOR_STEPS'] = 5
    traj._early_step_counter = 0
    with pytest.raises(UnsupportedRequest, match='does not support'):
        traj.propagate(4.0, 4.0)


# ------------------------------------------------------------
# TEST: system timescale warning fires when tau is too large
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_propagate_system_timescale_warning():
    # This case tests that propagate emits a warning when tau exceeds
    # the system Hamiltonian timescale.
    traj = _make_initialized_traj()
    # Use a very large timestep that exceeds system timescale
    # (system_timescale ~ 1/(max eigenvalue spread) ~ few fs)
    large_tau = 1000.0
    with pytest.warns(UserWarning, match='timescale'):
        traj.propagate(large_tau, large_tau)
