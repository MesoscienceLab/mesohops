import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp

__title__ = 'test_tensor_nonuniform_modes'
__author__ = 'A. Hartzell'
__maintainer__ = 'A. Hartzell'

# ============================================================
# Shared Setup: 3-state spectroscopy system
# ============================================================
# Ground state (index 0) has no bath coupling.
# Excited states (indices 1, 2) each have one L-operator with 2 modes.

n_site = 2
n_state = n_site + 1
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

H2_sys = np.zeros((n_state, n_state), dtype=np.complex128)
H2_sys[1:, 1:] = np.array([[100, -50], [-50, 0]], dtype=np.complex128)

# L-operators only for excited states (no ground-state bath coupling)
list_lop = []
list_gw_sysbath = []
for i in range(n_site):
    lop = np.zeros((n_state, n_state), dtype=np.float64)
    lop[i + 1, i + 1] = 1.0
    list_lop.append(sp.sparse.coo_matrix(lop))
    list_gw_sysbath.append([g_0, w_0])
    list_lop.append(lop)
    list_gw_sysbath.append([-1j * np.imag(g_0), 500.0])

sys_param = {
    'HAMILTONIAN': H2_sys,
    'GW_SYSBATH': list_gw_sysbath,
    'L_HIER': list_lop,
    'L_NOISE1': list_lop,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': list_gw_sysbath,
}

psi_0 = np.zeros(n_state, dtype=np.complex128)
psi_0[0] = 1.0

k_max = 2
state_list = np.arange(n_state)
delta_s = 0


def _make_spectroscopy_tensor(method):
    """Creates and initializes a HopsTensorWavefunction for the spectroscopy system."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    system = HopsSystem(sys_param)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(delta_s > 0, psi_0)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = state_list
    tb.initialize(delta_s)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    return ht


# ============================================================
# TEST SUITE: M1_modes_per_state computation
# ============================================================

# ------------------------------------------------------------
# TEST: non-uniform mode counts computed correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_modes_per_state_nonuniform():
    # This case tests that the ground state gets 0 modes and each excited
    # state gets the correct number of modes from LIST_HMODE_INDICES_BY_STATE
    ht = _make_spectroscopy_tensor('fullstate')
    # Ground state: 0 modes, excited states: 2 modes each (g_0/w_0 pair)
    expected = np.array([0, 2, 2])
    np.testing.assert_array_equal(ht.M1_modes_per_state, expected)


# ------------------------------------------------------------
# TEST: mode offset array is cumulative sum
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_mode_offset_nonuniform():
    # This case tests that M1_mode_offset is [0, cumsum(M1_modes_per_state)]
    ht = _make_spectroscopy_tensor('fullstate')
    expected = np.array([0, 0, 2, 4])
    np.testing.assert_array_equal(ht.M1_mode_offset, expected)


# ============================================================
# TEST SUITE: MPS core layout with non-uniform modes
# ============================================================

# ------------------------------------------------------------
# TEST: fullstate MPS has correct number of cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_fullstate_core_count_nonuniform():
    # This case tests that the MPS has 1 state core + total_modes mode cores
    ht = _make_spectroscopy_tensor('fullstate')
    # 1 state core + 4 mode cores (2 per excited state, 0 for ground)
    assert len(ht.list_cores_phi) == 5


# ------------------------------------------------------------
# TEST: phi_0 extraction works with non-uniform modes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_0_extraction_nonuniform():
    # This case tests that the physical wavefunction is recovered correctly
    ht = _make_spectroscopy_tensor('fullstate')
    V1_phi = ht.psi
    # Initial state is |g> = [1, 0, 0]
    np.testing.assert_allclose(V1_phi[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(V1_phi[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(V1_phi[2], 0.0, atol=1e-12)
