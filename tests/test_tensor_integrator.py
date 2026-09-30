import numpy as np
import pytest
import scipy as sp

from mesohops.integrator.tensor_integrator import (
    _build_tdvp_solver_kwargs,
    runge_kutta_step_tensor,
    runge_kutta_variables,
    single_point_variables,
    tdvp1_step_tensor,
    tdvp2_step_tensor,
)
from mesohops.noise.hops_noise import HopsNoise
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.tensor.hops_tensor_eom import HopsTensorEOM
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.physical_constants import hbar

__title__ = 'Unit Tests for Tensor Integrators'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'


# ============================================================
# Shared Setup: Noise
# ============================================================
# Mirrors test_integrator_rk.py noise configuration exactly.

noise_param = {
    'SEED': np.array([np.arange(-10, 10.5, 0.5), -1 * np.arange(-10, 10.5, 0.5)]),
    'MODEL': 'PRE_CALCULATED',
    'TLEN': 10.0,  # [fs]
    'TAU': 0.25,  # [fs]
}

noise_param_two = {
    'SEED': np.array(
        [np.arange(-10, 10.5, 0.5) / 2, -1 * np.arange(-10, 10.5, 0.5) / 2]
    ),
    'MODEL': 'PRE_CALCULATED',
    'TLEN': 10.0,  # [fs]
    'TAU': 0.25,  # [fs]
}

T3_loperator_noise = np.zeros([2, 2, 2], dtype=np.float64)
T3_loperator_noise[0, 0, 0] = 1.0
T3_loperator_noise[1, 1, 1] = 1.0

sys_param_noise = {
    'HAMILTONIAN': np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
    'GW_SYSBATH': [[10.0, 10.0], [5.0, 5.0]],
    'L_HIER': T3_loperator_noise,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': [[10.0, 10.0], [5.0, 5.0]],
    'L_NOISE1': T3_loperator_noise,
}

sys_param_noise['NSITE'] = len(sys_param_noise['HAMILTONIAN'][0])
sys_param_noise['NMODES'] = len(sys_param_noise['GW_SYSBATH'][0])
sys_param_noise['N_L2'] = 2
sys_param_noise['L_IND_BY_NMODE1'] = [0, 1]
sys_param_noise['LIND_DICT'] = {
    0: T3_loperator_noise[0, :, :],
    1: T3_loperator_noise[1, :, :],
}

noise_corr = {
    'CORR_FUNCTION': sys_param_noise['ALPHA_NOISE1'],
    'N_L2': sys_param_noise['N_L2'],
    'LIND_BY_NMODE': sys_param_noise['L_IND_BY_NMODE1'],
    'CORR_PARAM': sys_param_noise['PARAM_NOISE1'],
}


# ============================================================
# Shared Setup: Tensor System (4-site dimer of dimers)
# ============================================================

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

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0

k_max = 2
state_list = np.arange(nsite)
delta_s = 0
eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}


def _make_eom(method):
    """Returns (HopsTensorWavefunction, HopsTensorEOM, HopsTensorBasis), all initialized."""
    from mesohops.basis.hops_modes import HopsModes
    from mesohops.basis.hops_noise_memory import HopsNoiseMemory
    from mesohops.basis.hops_system import HopsSystem

    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    # Construct shared basis objects directly (same pattern as test_hops_tensor_basis)
    system = HopsSystem(sys_param)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    # Manually initialize shared objects (normally done by trajectory)
    system.initialize(delta_s > 0, psi_0)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = state_list
    tb.initialize(delta_s)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    eom = HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)
    return ht, eom, tb


# ============================================================
# TEST SUITE: runge_kutta_variables()
# ============================================================


# ------------------------------------------------------------
# TEST: Effective noise integration produces correct averages
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_variables_effective_noise_integration():
    # This case tests that runge_kutta_variables produces correct averaged
    # noise when effective_noise_integration=True, and point-sampled noise
    # when False. Mirrors test_integrator_rk.test_effective_noise_integration
    # but calls the tensor_integrator copy of runge_kutta_variables.
    test_noise = HopsNoise(noise_param, noise_corr)
    test_noise2 = HopsNoise(noise_param_two, noise_corr)

    rk_var_control = runge_kutta_variables(
        'b',
        5.0,
        test_noise,
        test_noise2,
        1.5,
        [0, 1],
        effective_noise_integration=False,
    )
    rk_var_integrated = runge_kutta_variables(
        'b',
        5.0,
        test_noise,
        test_noise2,
        1.5,
        [0, 1],
        effective_noise_integration=True,
    )

    # Noise fine-step index arithmetic: the noise arrays are sampled at
    # `noise_TAU = 0.25` resolution, so t=5.0 lands at index t / noise_TAU
    # = 20, and the span `[t, t + 1.5*tau)` with `tau = 1.5` advances
    # 1.5*1.5/0.25 = 9 fine steps to index 29 — but runge_kutta_variables
    # grabs an extra 3-step buffer for the half-step averaging, landing
    # the slice end at 20 + 12 = 32. Hence `[5*4 : 8*4]`.
    known_noise_1 = noise_param['SEED'][:, 5 * 4 : 8 * 4]
    known_control_1 = known_noise_1[:, np.array([0, 3, 6])]
    known_integrated_1 = np.array(
        [
            np.mean(known_noise_1[:, :3], axis=1),
            np.mean(known_noise_1[:, 3:6], axis=1),
            np.mean(known_noise_1[:, 6:9], axis=1),
        ]
    ).T

    known_noise_2 = noise_param_two['SEED'][:, 5 * 4 : 8 * 4]
    known_control_2 = known_noise_2[:, np.array([0, 3, 6])]
    known_integrated_2 = np.array(
        [
            np.mean(known_noise_2[:, :3], axis=1),
            np.mean(known_noise_2[:, 3:6], axis=1),
            np.mean(known_noise_2[:, 6:9], axis=1),
        ]
    ).T

    assert np.allclose(rk_var_control['z_rnd'], known_control_1)
    assert np.allclose(rk_var_control['z_rnd2'], known_control_2)
    assert np.allclose(rk_var_integrated['z_rnd'], known_integrated_1)
    assert np.allclose(rk_var_integrated['z_rnd2'], known_integrated_2)
    # Passthrough values: z_mem ('b') and tau (1.5) are forwarded
    # untouched regardless of the effective_noise_integration flag.
    assert rk_var_control['z_mem'] == 'b'
    assert rk_var_integrated['z_mem'] == 'b'
    assert rk_var_control['tau'] == 1.5
    assert rk_var_integrated['tau'] == 1.5


# ============================================================
# TEST SUITE: runge_kutta_step_tensor()
# ============================================================


# ------------------------------------------------------------
# TEST: RK4 step produces finite output and preserves norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_values():
    # CASE: Run a single RK4 step with a real EOM and verify the
    # output z_mem is finite and the wavefunction norm stays O(1).
    ht, eom, tb = _make_eom('fullstate')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 1.0
    z_mem = (0.01 + 0.02j) * np.arange(n_modes, dtype=np.complex128)
    z_rnd_3pt = (0.03 - 0.01j) * np.arange(
        n_l2,
        dtype=np.complex128,
    )
    z_rnd = np.column_stack(
        [z_rnd_3pt, z_rnd_3pt, z_rnd_3pt],
    )
    z_rnd2_3pt = (-0.02 + 0.04j) * np.arange(
        n_l2,
        dtype=np.complex128,
    )
    z_rnd2 = np.column_stack(
        [z_rnd2_3pt, z_rnd2_3pt, z_rnd2_3pt],
    )
    z_mem_rk = runge_kutta_step_tensor(
        eom,
        z_mem.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    # z_mem should be finite and changed from the input
    assert np.all(np.isfinite(z_mem_rk)), 'RK4 z_mem contains NaN or Inf'
    assert not np.allclose(z_mem_rk, z_mem), 'RK4 z_mem unchanged from input'
    # Wavefunction norm should be approximately preserved
    psi_after = eom.wavefunction.psi
    norm_after = np.linalg.norm(psi_after)
    assert 0.5 < norm_after < 2.0, (
        f'RK4 step produced extreme norm: {norm_after}'
    )


# ------------------------------------------------------------
# TEST: RK4 step runs with statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_statenumber():
    # CASE: Run a single RK4 step with statenumber representation.
    # The checkpoint copy has a nested-list branch for statenumber
    # that is not exercised by fullstate tests.
    ht, eom, tb = _make_eom('number')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 1.0
    z_mem = (0.01 + 0.02j) * np.arange(n_modes, dtype=np.complex128)
    z_rnd_3pt = (0.03 - 0.01j) * np.arange(
        n_l2,
        dtype=np.complex128,
    )
    z_rnd = np.column_stack(
        [z_rnd_3pt, z_rnd_3pt, z_rnd_3pt],
    )
    z_rnd2_3pt = (-0.02 + 0.04j) * np.arange(
        n_l2,
        dtype=np.complex128,
    )
    z_rnd2 = np.column_stack(
        [z_rnd2_3pt, z_rnd2_3pt, z_rnd2_3pt],
    )
    z_mem_rk = runge_kutta_step_tensor(
        eom,
        z_mem.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    assert np.all(np.isfinite(z_mem_rk)), 'RK4 z_mem contains NaN or Inf'
    assert not np.allclose(z_mem_rk, z_mem), 'RK4 z_mem unchanged from input'
    psi_after = eom.wavefunction.psi
    norm_after = np.linalg.norm(psi_after)
    assert 0.5 < norm_after < 2.0, (
        f'RK4 step produced extreme norm: {norm_after}'
    )


# ============================================================
# TEST SUITE: tdvp1_step_tensor()
# ============================================================


def _make_tb(sp=sys_param, ds=delta_s, psi=psi_0, sl=state_list):
    """Creates an initialized HopsTensorBasis from sys_param dict."""
    from mesohops.basis.hops_modes import HopsModes
    from mesohops.basis.hops_noise_memory import HopsNoiseMemory
    from mesohops.basis.hops_system import HopsSystem

    system = HopsSystem(sp)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(ds > 0, psi)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = sl
    tb.initialize(ds)
    return tb


def _make_eom_tdvp(method):
    """Returns (HopsTensorWavefunction, HopsTensorEOM, HopsTensorBasis) for TDVP1."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'TDVP1'}
    tb = _make_tb()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    eom = HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)
    return ht, eom, tb


def _make_eom_tdvp2(method):
    """Returns (HopsTensorWavefunction, HopsTensorEOM, HopsTensorBasis) for TDVP2."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'TDVP2'}
    tb = _make_tb()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    eom = HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)
    return ht, eom, tb


# ------------------------------------------------------------
# TEST: TDVP1 runs without error on fullstate representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tdvp1_step_tensor_runs_without_error():
    # This case tests that tdvp1_step_tensor runs to completion
    # on the fullstate representation.
    ht, eom, tb = _make_eom_tdvp('fullstate')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 0.5
    z_mem = np.zeros(n_modes, dtype=np.complex128)
    z_rnd = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    z_rnd2 = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    psi_before = eom.wavefunction.psi.copy()
    norm_before = np.linalg.norm(psi_before)
    z_mem_out = tdvp1_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau)
    assert z_mem_out.shape == z_mem.shape
    assert not np.any(np.isnan(z_mem_out))
    # Invariant: TDVP approximately preserves norm
    norm_after = np.linalg.norm(eom.wavefunction.psi)
    np.testing.assert_allclose(
        norm_after, norm_before, atol=0.1,
        err_msg='TDVP step should approximately preserve norm',
    )


# ------------------------------------------------------------
# TEST: TDVP1 updates phi (MPS cores change after step)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tdvp1_step_tensor_updates_phi():
    # This case tests that tdvp1_step_tensor actually modifies
    # the MPS wavefunction.
    ht, eom, tb = _make_eom_tdvp('fullstate')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 0.5
    z_mem = np.zeros(n_modes, dtype=np.complex128)
    z_rnd = np.column_stack(
        [
            0.01 * np.arange(n_l2, dtype=np.complex128),
        ]
    )
    z_rnd2 = np.column_stack(
        [
            0.01 * np.arange(n_l2, dtype=np.complex128),
        ]
    )
    cores_before = [c.copy() for c in ht.list_cores_phi]
    tdvp1_step_tensor(eom, z_mem.copy(), z_rnd, z_rnd2, tau)
    cores_after = ht.list_cores_phi
    # Shapes may change (TDVP re-canonicalizes), so compare contracted MPS
    psi_before = cores_before[0]
    for c in cores_before[1:]:
        psi_before = np.tensordot(psi_before, c, axes=([-1], [0]))
    psi_after = cores_after[0]
    for c in cores_after[1:]:
        psi_after = np.tensordot(psi_after, c, axes=([-1], [0]))
    assert not np.allclose(psi_before, psi_after), (
        'TDVP1 step did not modify the MPS wavefunction'
    )


# ------------------------------------------------------------
# TEST: TDVP1 on statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tdvp1_step_tensor_statenumber():
    # This case tests that tdvp1_step_tensor works with
    # statenumber representation.
    ht, eom, tb = _make_eom_tdvp('number')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 0.5
    z_mem = np.zeros(n_modes, dtype=np.complex128)
    z_rnd = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    z_rnd2 = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    psi_before = eom.wavefunction.psi.copy()
    norm_before = np.linalg.norm(psi_before)
    z_mem_out = tdvp1_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau)
    assert z_mem_out.shape == z_mem.shape
    assert not np.any(np.isnan(z_mem_out))
    # Invariant: TDVP approximately preserves norm
    norm_after = np.linalg.norm(eom.wavefunction.psi)
    np.testing.assert_allclose(
        norm_after, norm_before, atol=0.1,
        err_msg='TDVP step should approximately preserve norm',
    )


# ============================================================
# TEST SUITE: tdvp2_step_tensor()
# ============================================================


# ------------------------------------------------------------
# TEST: TDVP2 runs without error on fullstate representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tdvp2_step_tensor_runs_without_error():
    # This case tests that tdvp2_step_tensor runs to completion
    # on the fullstate representation.
    ht, eom, tb = _make_eom_tdvp2('fullstate')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 0.5
    z_mem = np.zeros(n_modes, dtype=np.complex128)
    z_rnd = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    z_rnd2 = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    psi_before = eom.wavefunction.psi.copy()
    norm_before = np.linalg.norm(psi_before)
    z_mem_out = tdvp2_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau)
    assert z_mem_out.shape == z_mem.shape
    assert not np.any(np.isnan(z_mem_out))
    # Invariant: TDVP approximately preserves norm
    norm_after = np.linalg.norm(eom.wavefunction.psi)
    np.testing.assert_allclose(
        norm_after, norm_before, atol=0.1,
        err_msg='TDVP step should approximately preserve norm',
    )


# ------------------------------------------------------------
# TEST: TDVP2 on statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tdvp2_step_tensor_statenumber():
    # This case tests that tdvp2_step_tensor works with
    # statenumber representation (non-uniform physical dims).
    ht, eom, tb = _make_eom_tdvp2('number')
    n_modes = len(tb.mode.list_modeidx_abs)
    n_l2 = tb.mode.n_l2
    tau = 0.5
    z_mem = np.zeros(n_modes, dtype=np.complex128)
    z_rnd = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    z_rnd2 = np.column_stack(
        [
            np.zeros(n_l2, dtype=np.complex128),
        ]
    )
    psi_before = eom.wavefunction.psi.copy()
    norm_before = np.linalg.norm(psi_before)
    z_mem_out = tdvp2_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau)
    assert z_mem_out.shape == z_mem.shape
    assert not np.any(np.isnan(z_mem_out))
    # Invariant: TDVP approximately preserves norm
    norm_after = np.linalg.norm(eom.wavefunction.psi)
    np.testing.assert_allclose(
        norm_after, norm_before, atol=0.1,
        err_msg='TDVP step should approximately preserve norm',
    )


# ------------------------------------------------------------
# TEST: TDVP1 and TDVP2 z_mem agree (both Euler z_mem update)
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tdvp1_tdvp2_zmem_agree():
    # This case tests that TDVP1 and TDVP2 produce the same
    # z_mem update (both use Euler), verifying the z_mem path
    # is identical.
    ht1, eom1, tb1 = _make_eom_tdvp('fullstate')
    ht2, eom2, tb2 = _make_eom_tdvp2('fullstate')
    n_modes = len(tb1.mode.list_modeidx_abs)
    n_l2 = tb1.mode.n_l2
    tau = 0.5
    z_mem = np.zeros(n_modes, dtype=np.complex128)
    z_rnd = np.column_stack(
        [
            0.01 * np.arange(n_l2, dtype=np.complex128),
        ]
    )
    z_rnd2 = np.column_stack(
        [
            -0.005 * np.arange(n_l2, dtype=np.complex128),
        ]
    )
    z_mem_1 = tdvp1_step_tensor(eom1, z_mem.copy(), z_rnd, z_rnd2, tau)
    z_mem_2 = tdvp2_step_tensor(eom2, z_mem.copy(), z_rnd, z_rnd2, tau)
    np.testing.assert_allclose(
        z_mem_1,
        z_mem_2,
        atol=1e-10,
        err_msg='TDVP1 and TDVP2 z_mem should agree (both Euler update)',
    )


# ============================================================
# TEST SUITE: runge_kutta_step_tensor() — RK4 formula verification
# ============================================================


def _contract_mps_to_vector(list_cores):
    """Contract a list of MPS cores into a single state vector."""
    state = list_cores[0]
    for core in list_cores[1:]:
        state = np.tensordot(state, core, axes=([-1], [0]))
    return state.reshape(-1)


class _MockPhiTensor:
    """Minimal mock for HopsTensorWavefunction with 2-core bond-dim-1 MPS.

    Uses a trivial scalar first core (value 1) and a second core encoding
    the physical state. The bond between the two cores is always rank-1
    after tensor_add compression, so OBC is preserved and the mock
    remains analytically tractable.
    """

    __slots__ = ('list_cores_phi', 'mps_epsilon', 'bond_dim_max')

    def __init__(self, n_phys):
        # Core 0: trivial scalar 1; core 1: physical state (initially zero)
        self.list_cores_phi = [
            np.ones((1, 1, 1), dtype=np.complex128),
            np.zeros((1, n_phys, 1), dtype=np.complex128),
        ]
        self.mps_epsilon = 1e-10
        self.bond_dim_max = 20

    @property
    def flat_cores(self):
        return self.list_cores_phi

    def restore_phi(self, cores):
        self.list_cores_phi = [c.copy() for c in cores]

    def update_phi_from_flat(self, cores):
        self.list_cores_phi = [c.copy() for c in cores]


class _MockEOM:
    """Mock EOM that returns constant derivatives.

    build_generator always returns dz = V1_dz_const.
    derivative always returns a 2-core MPS encoding V1_dphi_const.
    This makes the RK4 output analytically predictable.
    """

    __slots__ = (
        'wavefunction', 'V1_dphi_const', 'V1_dz_const',
        # Side-channel attributes the real HopsTensorEOM publishes;
        # integrator tests read these to track complexity across stages.
        'last_matvec_complexity', 'max_complexity_step',
    )

    def __init__(self, n_phys, n_zmem):
        self.wavefunction = _MockPhiTensor(n_phys)
        # Constant derivatives with different complex values to
        # break symmetry and catch index/weighting errors
        self.V1_dphi_const = (0.1 + 0.2j) * np.arange(
            1, n_phys + 1, dtype=np.complex128
        )
        self.V1_dz_const = (0.3 - 0.1j) * np.arange(1, n_zmem + 1, dtype=np.complex128)
        # Match the HopsTensorEOM side-channel contract; tests may
        # override `last_matvec_complexity` inside `derivative` to pin
        # the RK4 max-tracking behavior.
        self.last_matvec_complexity = 0
        self.max_complexity_step = 0

    def build_generator(self, z_mem, z_rnd, z_rnd2):
        return self.V1_dz_const.copy()

    def derivative(self):
        # Side-channel: real derivative() sets this; mock sets 0 as a
        # sentinel so max-tracking tests can override via subclassing.
        self.last_matvec_complexity = 0
        return [
            np.ones((1, 1, 1), dtype=np.complex128),
            self.V1_dphi_const.reshape(1, -1, 1).copy(),
        ]


# ------------------------------------------------------------
# TEST: RK4 z_mem matches analytical formula with constant dz
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_zmem_formula():
    # With constant dz at every stage, the RK4 z_mem formula reduces to:
    #   z_new = z_old + tau/hbar * dz * (1/6 + 2/6 + 2/6 + 1/6)
    #         = z_old + tau/hbar * dz
    # because the weights sum to 1.
    n_phys = 3
    n_zmem = 2
    mock_eom = _MockEOM(n_phys, n_zmem)
    tau = 2.0
    z_mem_0 = (0.05 + 0.01j) * np.arange(n_zmem, dtype=np.complex128)
    # z_rnd shape: (n_modes, 3) — 3 time points for RK4
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    z_mem_new = runge_kutta_step_tensor(
        mock_eom,
        z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    # Constant dz → weights (1+2+2+1)/6 = 1 → z_new = z_old + tau/hbar * dz
    z_mem_expected = z_mem_0 + tau / hbar * mock_eom.V1_dz_const
    np.testing.assert_allclose(
        z_mem_new,
        z_mem_expected,
        atol=1e-12,
        err_msg='RK4 z_mem does not match formula for constant dz',
    )


# ------------------------------------------------------------
# TEST: RK4 phi matches analytical formula with constant dphi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_phi_formula():
    # With constant dphi at every stage, the RK4 phi formula is:
    #   phi_new = phi_old + (tau/hbar) * dphi
    #             * (1/6 + 1/3 + 1/3 + 1/6)
    #           = phi_old + (tau/hbar) * dphi
    # The final phi lives in mock_eom.wavefunction.list_cores_phi.
    # For bond-dim-1 single-core MPS, this is exact (no SVD truncation).
    n_phys = 3
    n_zmem = 2
    mock_eom = _MockEOM(n_phys, n_zmem)
    tau = 2.0
    # Set initial phi to a known state
    V1_phi_0 = (1.0 + 0.5j) * np.arange(1, n_phys + 1, dtype=np.complex128)
    mock_eom.wavefunction.list_cores_phi = [
        np.ones((1, 1, 1), dtype=np.complex128),
        V1_phi_0.reshape(1, -1, 1).copy(),
    ]
    z_mem_0 = np.zeros(n_zmem, dtype=np.complex128)
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    runge_kutta_step_tensor(
        mock_eom,
        z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    # Contract the MPS to a state vector
    V1_phi_new = _contract_mps_to_vector(mock_eom.wavefunction.list_cores_phi)
    # Constant dphi → weights sum to 1 →
    #   phi_new = phi_old + (tau/hbar) * dphi
    V1_phi_expected = V1_phi_0 + tau / hbar * mock_eom.V1_dphi_const
    np.testing.assert_allclose(
        V1_phi_new,
        V1_phi_expected,
        atol=1e-10,
        err_msg='RK4 phi does not match formula for constant dphi',
    )


# ------------------------------------------------------------
# TEST: RK4 resets phi to original state before each stage
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_state_reset():
    # This test verifies that each RK4 stage starts from the
    # original phi, not from a mutated intermediate. We use a
    # mock EOM that records the phi it sees at each build_generator
    # call. With constant derivatives, stages 1-3 should see
    # phi_0 + c_rk[i] * k[i-1] (scaled), but the key check is
    # that after the full step, phi is built from phi_0 + weighted sum
    # (not accumulated from intermediates).
    n_phys = 2
    n_zmem = 1

    # Track what phi the EOM sees at each stage
    list_phi_seen = []

    class _RecordingEOM(_MockEOM):
        __slots__ = ()

        def build_generator(self, z_mem, z_rnd, z_rnd2):
            # Record the current phi state by contracting MPS
            V1_phi_cur = _contract_mps_to_vector(
                self.wavefunction.list_cores_phi,
            )
            list_phi_seen.append(V1_phi_cur.copy())
            return self.V1_dz_const.copy()

    mock_eom = _RecordingEOM(n_phys, n_zmem)
    V1_phi_0 = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    mock_eom.wavefunction.list_cores_phi = [
        np.ones((1, 1, 1), dtype=np.complex128),
        V1_phi_0.reshape(1, -1, 1).copy(),
    ]
    tau = 1.0
    z_mem_0 = np.zeros(n_zmem, dtype=np.complex128)
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    runge_kutta_step_tensor(
        mock_eom,
        z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    # Stage 0 (i=0): should see phi_0 unchanged
    np.testing.assert_allclose(
        list_phi_seen[0],
        V1_phi_0,
        atol=1e-12,
        err_msg='Stage 0 did not start from original phi',
    )
    # Stages 1-3 (i=1,2,3): should see phi_0 + c_rk[i]*k[i-1]*(tau/hbar)
    # not an accumulated state from previous stages
    c_rk = [0.0, 0.5, 0.5, 1.0]
    for stage in range(1, 4):
        V1_phi_expected = (
            V1_phi_0 + c_rk[stage] * tau / hbar * mock_eom.V1_dphi_const
        )
        np.testing.assert_allclose(
            list_phi_seen[stage],
            V1_phi_expected,
            atol=1e-10,
            err_msg=f'Stage {stage} did not reset to phi_0 + c*k',
        )


# ------------------------------------------------------------
# TEST: RK4 weights are individually correct (state-dependent derivative)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_rk4_weights():
    # CASE: With dphi/dt = phi (derivative returns current state),
    # the RK4 formula for one step reduces to the 4th-order Taylor
    # expansion of exp(a) where a = tau/hbar:
    #   phi_new = phi_0 * (1 + a + a^2/2 + a^3/6 + a^4/24)
    # If any individual RK4 weight (1/6, 1/3, 1/3, 1/6) were wrong,
    # the polynomial coefficients would differ. This test catches
    # weight errors that the constant-derivative tests miss.
    n_phys = 2
    n_zmem = 1

    class _StateDependentEOM(_MockEOM):
        __slots__ = ()

        def derivative(self):
            self.last_matvec_complexity = 0
            return [c.copy() for c in self.wavefunction.list_cores_phi]

    mock_eom = _StateDependentEOM(n_phys, n_zmem)
    V1_phi_0 = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    mock_eom.wavefunction.list_cores_phi = [
        np.ones((1, 1, 1), dtype=np.complex128),
        V1_phi_0.reshape(1, -1, 1).copy(),
    ]
    tau = 1.0
    z_mem_0 = np.zeros(n_zmem, dtype=np.complex128)
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    runge_kutta_step_tensor(
        mock_eom,
        z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    V1_phi_new = _contract_mps_to_vector(mock_eom.wavefunction.list_cores_phi)
    # RK4 applied to dphi/dt = phi gives the 4th-order exponential Taylor series
    a = tau / hbar
    V1_phi_expected = V1_phi_0 * (1 + a + a**2 / 2 + a**3 / 6 + a**4 / 24)
    np.testing.assert_allclose(
        V1_phi_new,
        V1_phi_expected,
        atol=1e-10,
        err_msg='RK4 weights are incorrect (state-dependent derivative test)',
    )


# ------------------------------------------------------------
# TEST: RK4 z_mem weights are individually correct (z-dependent generator)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_rk4_weights_zmem():
    # CASE: With dz/dt = z (build_generator returns current z_mem),
    # the RK4 formula for one step reduces to the 4th-order Taylor
    # expansion of exp(a) where a = tau/hbar. This pins the individual
    # z_mem weights (1, 2, 2, 1)/6 on the z_mem path (source L122-133)
    # independently of the phi path — a constant-dz test passes for
    # any weight set summing to 1, so it can't detect wrong individual
    # weights here.
    n_phys = 2
    n_zmem = 2

    class _ZmemDependentEOM(_MockEOM):
        __slots__ = ()

        def build_generator(self, z_mem, z_rnd, z_rnd2):
            # dz/dt = z — the z-analog of the state-dependent phi test
            return z_mem.copy()

    mock_eom = _ZmemDependentEOM(n_phys, n_zmem)
    V1_z_mem_0 = np.array([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex128)
    tau = 1.0
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    z_mem_new = runge_kutta_step_tensor(
        mock_eom,
        V1_z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    # RK4 applied to dz/dt = z gives the 4th-order exponential Taylor series
    a = tau / hbar
    V1_z_mem_expected = V1_z_mem_0 * (
        1 + a + a**2 / 2 + a**3 / 6 + a**4 / 24
    )
    np.testing.assert_allclose(
        z_mem_new,
        V1_z_mem_expected,
        atol=1e-10,
        err_msg='RK4 z_mem weights are incorrect (z-dependent generator test)',
    )


# ------------------------------------------------------------
# TEST: RK4 publishes max complexity across the four substeps
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_max_complexity_across_substeps():
    # This case tests that runge_kutta_step_tensor publishes the MAX of
    # the four per-substep complexities on eom.max_complexity_step, not
    # the first/last/sum/etc. With derivative reporting [5, 10, 3, 7]
    # across the four RK4 stages, the published max must be 10. The
    # constant-complexity mocks elsewhere always report 0, so this test
    # is the only one that pins the max-tracking semantics.
    n_phys = 2
    n_zmem = 1

    class _MaxTrackingEOM(_MockEOM):
        __slots__ = ('list_complexities', '_call_count')

        def __init__(self, n_phys, n_zmem, list_complexities):
            super().__init__(n_phys, n_zmem)
            self.list_complexities = list_complexities
            self._call_count = 0

        def derivative(self):
            cores = super().derivative()
            self.last_matvec_complexity = (
                self.list_complexities[self._call_count]
            )
            self._call_count += 1
            return cores

    list_complexities = [5, 10, 3, 7]
    mock_eom = _MaxTrackingEOM(n_phys, n_zmem, list_complexities)
    tau = 1.0
    z_mem_0 = np.zeros(n_zmem, dtype=np.complex128)
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    runge_kutta_step_tensor(mock_eom, z_mem_0, z_rnd, z_rnd2, tau)
    assert mock_eom._call_count == 4, (
        'RK4 should call derivative exactly 4 times'
    )
    assert mock_eom.max_complexity_step == max(list_complexities), (
        f'Expected max_complexity_step={max(list_complexities)}, '
        f'got {mock_eom.max_complexity_step}'
    )


# ------------------------------------------------------------
# TEST: TDVP publishes max_complexity_step = 0
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tdvp_step_tensor_max_complexity_zero():
    # This case tests that tdvp_step_tensor unconditionally publishes 0
    # on eom.max_complexity_step. TDVP doesn't call tensor_matvec_prod,
    # so the peak-size proxy doesn't apply; the contract is "0 always".
    # Stuffing a non-zero value into eom.max_complexity_step before the
    # call lets us verify the step function actually overwrites — a no-op
    # that simply leaves the previous value untouched would silently
    # leak data from the previous step.
    _, eom, _ = _make_eom_tdvp(method='fullstate')
    eom.max_complexity_step = 999  # sentinel — must be overwritten
    n_zmem = len(eom.noise_memory.list_zmemmodeidx_abs)
    z_mem = np.zeros(n_zmem, dtype=np.complex128)
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)
    eom.build_generator(
        z_mem, z_rnd[:, 0], z_rnd2[:, 0],
    )  # populate eom.mpo_cores
    tdvp1_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau=1.0)
    assert eom.max_complexity_step == 0, (
        f'TDVP should publish max_complexity_step=0, '
        f'got {eom.max_complexity_step}'
    )


# ------------------------------------------------------------
# TEST: RK4 phi formula with statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_phi_formula_statenumber():
    # This case tests that the RK4 phi update works correctly with
    # statenumber (nested list-of-lists) MPS, exercising the checkpoint
    # deep-copy branch (isinstance(g, list)) and the nested-aware
    # tensor_add/scale_mps paths. With constant dphi, the formula is
    # the same as fullstate: phi_new = phi_old + (tau/hbar) * dphi.
    n_state = 2
    n_zmem = 1
    k_max = 1  # mode core phys dim = k_max + 1 = 2

    # Statenumber MPS: each state group has [state_core, mode_core]
    # State core shape: (1, 2, 1) — binary occupied/unoccupied
    # Mode core shape: (1, 2, 1) — k=0,1 occupation
    state_0 = np.zeros((1, 2, 1), dtype=np.complex128)
    state_0[0, 1, 0] = 1.0  # state 0 occupied
    mode_0 = np.zeros((1, 2, 1), dtype=np.complex128)
    mode_0[0, 0, 0] = 1.0  # k=0

    state_1 = np.zeros((1, 2, 1), dtype=np.complex128)
    state_1[0, 0, 0] = 1.0  # state 1 unoccupied
    mode_1 = np.zeros((1, 2, 1), dtype=np.complex128)
    mode_1[0, 0, 0] = 1.0  # k=0

    list_cores_phi_0 = [[state_0, mode_0], [state_1, mode_1]]

    # Constant derivative MPS with same nested structure
    dstate_0 = np.zeros((1, 2, 1), dtype=np.complex128)
    dstate_0[0, 1, 0] = 0.1 + 0.2j
    dmode_0 = np.zeros((1, 2, 1), dtype=np.complex128)
    dmode_0[0, 0, 0] = 1.0

    dstate_1 = np.zeros((1, 2, 1), dtype=np.complex128)
    dstate_1[0, 0, 0] = 0.0  # unoccupied stays zero
    dmode_1 = np.zeros((1, 2, 1), dtype=np.complex128)
    dmode_1[0, 0, 0] = 1.0

    list_cores_dphi = [[dstate_0, dmode_0], [dstate_1, dmode_1]]

    class _MockPhiStatenumber:
        __slots__ = ('list_cores_phi', 'mps_epsilon', 'bond_dim_max')

        def __init__(self):
            self.list_cores_phi = [
                [c.copy() for c in g] for g in list_cores_phi_0
            ]
            self.mps_epsilon = 1e-10
            self.bond_dim_max = 20

        @property
        def flat_cores(self):
            return [c for g in self.list_cores_phi for c in g]

        def restore_phi(self, cores):
            self.list_cores_phi = [
                [c.copy() for c in g] for g in cores
            ]

        def update_phi_from_flat(self, cores):
            self.list_cores_phi = [
                [c.copy() for c in g] for g in cores
            ]

    class _MockEOMStatenumber:
        __slots__ = (
            'wavefunction', 'V1_dz_const',
            'last_matvec_complexity', 'max_complexity_step',
        )

        def __init__(self):
            self.wavefunction = _MockPhiStatenumber()
            self.V1_dz_const = np.array([0.3 - 0.1j], dtype=np.complex128)
            self.last_matvec_complexity = 0
            self.max_complexity_step = 0

        def build_generator(self, z_mem, z_rnd, z_rnd2):
            return self.V1_dz_const.copy()

        def derivative(self):
            self.last_matvec_complexity = 0
            return [[c.copy() for c in g] for g in list_cores_dphi]

    mock_eom = _MockEOMStatenumber()
    tau = 2.0
    z_mem_0 = np.zeros(n_zmem, dtype=np.complex128)
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    # Record psi before
    from mesohops.util.tensor_operations import extract_psi
    V1_psi_before = extract_psi(
        mock_eom.wavefunction.list_cores_phi,
        'number',
        np.array([1, 1]),
    )

    runge_kutta_step_tensor(
        mock_eom,
        z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )

    V1_psi_after = extract_psi(
        mock_eom.wavefunction.list_cores_phi,
        'number',
        np.array([1, 1]),
    )

    # Constant dphi → weights sum to 1 →
    #   psi_new = psi_old + (tau/hbar) * dpsi_const
    V1_dpsi = extract_psi(
        list_cores_dphi,
        'number',
        np.array([1, 1]),
    )
    V1_psi_expected = V1_psi_before + tau / hbar * V1_dpsi
    np.testing.assert_allclose(
        V1_psi_after,
        V1_psi_expected,
        atol=1e-10,
        err_msg='RK4 phi formula failed for statenumber representation',
    )


# ------------------------------------------------------------
# TEST: RK4 stages receive correct noise time-point indices
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_noise_indexing():
    # CASE: Verify that RK4 stages 0-3 receive noise columns
    # [0, 1, 1, 2] (corresponding to t, t+tau/2, t+tau/2, t+tau).
    n_phys = 2
    n_zmem = 1
    list_z_rnd_seen = []
    list_z_rnd2_seen = []

    class _NoiseRecordingEOM(_MockEOM):
        __slots__ = ()

        def build_generator(self, z_mem, z_rnd, z_rnd2):
            list_z_rnd_seen.append(z_rnd.copy())
            list_z_rnd2_seen.append(z_rnd2.copy())
            return self.V1_dz_const.copy()

    mock_eom = _NoiseRecordingEOM(n_phys, n_zmem)
    tau = 1.0
    z_mem_0 = np.zeros(n_zmem, dtype=np.complex128)
    # Distinct values per column to identify which was passed
    z_rnd = np.array([[1.0 + 0j, 2.0 + 0j, 3.0 + 0j]])
    z_rnd2 = np.array([[4.0 + 0j, 5.0 + 0j, 6.0 + 0j]])

    runge_kutta_step_tensor(
        mock_eom,
        z_mem_0.copy(),
        z_rnd,
        z_rnd2,
        tau,
    )
    # Stages should receive noise columns: [0, 1, 1, 2]
    list_expected_col = [0, 1, 1, 2]
    for i in range(4):
        np.testing.assert_array_equal(
            list_z_rnd_seen[i], z_rnd[:, list_expected_col[i]],
            err_msg=f'Stage {i} received wrong z_rnd noise column',
        )
        np.testing.assert_array_equal(
            list_z_rnd2_seen[i], z_rnd2[:, list_expected_col[i]],
            err_msg=f'Stage {i} received wrong z_rnd2 noise column',
        )


# ------------------------------------------------------------
# TEST: RK4 does not mutate the input z_mem array
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_runge_kutta_step_tensor_zmem_not_mutated():
    # CASE: The input z_mem array should not be modified in-place.
    # At i=0 the code aliases z_mem_tmp = z_mem (not a copy), so
    # this test guards against in-place mutation by build_generator
    # or the final z_mem update using += instead of +.
    n_phys = 3
    n_zmem = 2
    mock_eom = _MockEOM(n_phys, n_zmem)
    tau = 2.0
    z_mem_0 = (0.05 + 0.01j) * np.arange(n_zmem, dtype=np.complex128)
    z_mem_input = z_mem_0.copy()
    z_rnd = np.zeros((n_zmem, 3), dtype=np.complex128)
    z_rnd2 = np.zeros((n_zmem, 3), dtype=np.complex128)

    runge_kutta_step_tensor(
        mock_eom,
        z_mem_input,
        z_rnd,
        z_rnd2,
        tau,
    )
    np.testing.assert_array_equal(
        z_mem_input, z_mem_0,
        err_msg='runge_kutta_step_tensor mutated the input z_mem array',
    )


# ============================================================
# TEST SUITE: _build_tdvp_solver_kwargs()
# ============================================================


# ------------------------------------------------------------
# TEST: krylov alias resolves to arnoldi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_tdvp_solver_kwargs_krylov_alias():
    # The 'krylov' update_type is a convenience alias for 'arnoldi'.
    # Verify that the returned dict has solver='arnoldi' and the
    # correct conv_tol.
    result = _build_tdvp_solver_kwargs('krylov', 1e-4)
    assert result['solver'] == 'arnoldi'
    assert result['conv_tol'] == 1e-4


# ------------------------------------------------------------
# TEST: lanczos returns correct kwargs
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_tdvp_solver_kwargs_lanczos():
    # 'lanczos' is passed through unchanged with conv_tol set.
    result = _build_tdvp_solver_kwargs('lanczos', 1e-6)
    assert result['solver'] == 'lanczos'
    assert result['conv_tol'] == 1e-6


# ------------------------------------------------------------
# TEST: ivp returns method, rtol, atol, and optional max_step
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_tdvp_solver_kwargs_ivp():
    # With all ivp kwargs provided, max_step is included.
    result = _build_tdvp_solver_kwargs(
        'ivp', 1e-6, ivp_method='RK45', ivp_rtol=1e-5,
        ivp_atol=1e-7, ivp_max_step=0.01,
    )
    assert result['solver'] == 'ivp'
    assert result['method'] == 'RK45'
    assert result['rtol'] == 1e-5
    assert result['atol'] == 1e-7
    assert result['max_step'] == 0.01
    # When ivp_max_step is not provided, max_step must be absent.
    result_no_max = _build_tdvp_solver_kwargs('ivp', 1e-6)
    assert 'max_step' not in result_no_max


# ============================================================
# TEST SUITE: single_point_variables()
# ============================================================


# ------------------------------------------------------------
# TEST: single_point_variables raises NotImplementedError for effective noise
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_single_point_variables_effective_noise_raises():
    # effective_noise_integration is not implemented for single_point_variables;
    # verify it raises NotImplementedError.
    test_noise = HopsNoise(noise_param, noise_corr)
    test_noise2 = HopsNoise(noise_param_two, noise_corr)
    with pytest.raises(NotImplementedError):
        single_point_variables(
            None,
            0.0,
            test_noise,
            test_noise2,
            0.5,
            effective_noise_integration=True,
        )


# ------------------------------------------------------------
# TEST: single_point_variables returns correct dict structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_single_point_variables_normal_path():
    # This case tests that single_point_variables returns a dict
    # with keys z_mem, z_rnd, z_rnd2, tau and that the noise arrays
    # have shape (n_l2, 1) — one time point sampled per L2 operator.
    test_noise = HopsNoise(noise_param, noise_corr)
    test_noise2 = HopsNoise(noise_param_two, noise_corr)
    n_l2 = sys_param_noise['N_L2']
    z_mem_in = np.array([0.1 + 0.2j, -0.3 + 0.0j])
    tau = 0.5
    t = 1.0

    result = single_point_variables(
        z_mem_in,
        t,
        test_noise,
        test_noise2,
        tau,
        effective_noise_integration=False,
    )

    # Returned dict must have exactly these keys
    assert set(result.keys()) == {'z_mem', 'z_rnd', 'z_rnd2', 'tau'}
    # z_mem is passed through unchanged
    np.testing.assert_array_equal(result['z_mem'], z_mem_in)
    # tau is passed through unchanged
    assert result['tau'] == tau
    # Noise arrays: shape (n_l2, 1) — one time point
    assert result['z_rnd'].shape == (n_l2, 1)
    assert result['z_rnd2'].shape == (n_l2, 1)
