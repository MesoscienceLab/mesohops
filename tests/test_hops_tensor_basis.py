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

__title__ = 'Unit Tests for HopsTensorBasis'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'

# ============================================================
# Shared Setup
# ============================================================

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
    'HAMILTONIAN': np.array(hs, dtype=np.complex128),
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': lop_list,
    'L_NOISE1': lop_list,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': gw_sysbath,
}

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0

state_list = list(np.arange(nsite))

k_max = 4

tensor_param = {
    'MPS_EPSILON': 1e-10,
    'METHOD': 'fullstate',
    'BOND_DIM_MAX': 20,
}

integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}

eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}


class _MockEOM:
    """Minimal mock for HopsTensorEOM with a no-op refresh_builder."""

    def refresh_builder(self):
        pass


def _make_basis_objects(sp=sys_param):
    '''Creates HopsSystem, HopsModes, HopsNoiseMemory directly.'''
    system = HopsSystem(sp)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    return system, mode, noise_memory


def _make_tensor_basis():
    '''Creates a HopsTensorBasis with directly-constructed objects.'''
    system, mode, noise_memory = _make_basis_objects()
    return HopsTensorBasis(system, mode, noise_memory)


def _make_initialized_tensor_basis(delta_s=0, sl=None, psi=None):
    '''Creates and initializes a HopsTensorBasis.'''
    if sl is None:
        sl = state_list
    if psi is None:
        psi = psi_0
    system, mode, noise_memory = _make_basis_objects()
    tb = HopsTensorBasis(system, mode, noise_memory)
    # Manually initialize the shared objects (normally done by trajectory)
    system.initialize(delta_s > 0, psi)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = sl
    tb.initialize(delta_s)
    return tb


def _make_tensor_for_tb(tb, method='fullstate'):
    '''Creates and initializes a HopsTensorWavefunction tied to a basis.'''
    tp = dict(tensor_param, METHOD=method)
    ht = HopsTensorWavefunction(k_max, tp, integrator_param, eom_param)
    ht.initialize(
        psi_0[tb.system.state_list],
        tb.system,
    )
    return ht


# ============================================================
# TEST SUITE: HopsTensorBasis()
# ============================================================


# ------------------------------------------------------------
# TEST: Constructor stores references and defaults
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_stores_references():
    '''HopsTensorBasis constructor stores refs and sets defaults.'''
    # This case tests that __init__ stores system, mode, noise_memory refs
    system = HopsSystem(sys_param)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    assert tb.system is system
    assert tb.mode is mode
    assert tb.noise_memory is noise_memory
    # This case tests that defaults are set correctly
    assert tb.eom is None
    assert tb.adaptive is False
    assert tb.delta_s == 0


# ------------------------------------------------------------
# TEST: initialize() with delta_s=0 sets non-adaptive
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_nonadaptive():
    '''initialize with delta_s=0 leaves adaptive=False.'''
    # Analytical: delta_s=0 means non-adaptive
    tb = _make_initialized_tensor_basis()
    assert tb.adaptive is False
    assert tb.delta_s == 0


# ------------------------------------------------------------
# TEST: initialize() with delta_s>0 sets adaptive
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_adaptive():
    '''initialize with delta_s>0 sets adaptive=True.'''
    # Analytical: delta_s > 0 means adaptive
    tb = _make_initialized_tensor_basis(delta_s=0.1)
    assert tb.adaptive is True
    assert tb.delta_s == 0.1


# ------------------------------------------------------------
# TEST: define_basis non-adaptive returns empty lists
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_define_basis_nonadaptive_returns_empty():
    '''define_basis returns ([], []) when not adaptive.'''
    # Analytical: non-adaptive define_basis returns empty lists immediately
    tb = _make_initialized_tensor_basis(delta_s=0)
    ht = _make_tensor_for_tb(tb)
    z_step = [np.zeros(1)] * len(tb.mode.list_modeidx_abs)
    list_old, list_new = tb.define_basis(ht, z_step)
    assert list_old == []
    assert list_new == []


# ------------------------------------------------------------
# TEST: update_basis adding a state grows state_list
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_update_basis_add_grows_state_list():
    '''update_basis with list_states_new grows system.state_list.'''
    # Setup: 3-state basis [0, 1, 2], add state 3
    sl_initial = [0, 1, 2]
    psi_3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
    tb = _make_initialized_tensor_basis(sl=sl_initial, psi=psi_3)
    ht = _make_tensor_for_tb(tb)
    tb.eom = _MockEOM()
    z_mem = np.zeros(len(tb.mode.list_modeidx_abs), dtype=np.complex128)
    state_list_before = sorted(tb.system.state_list)
    # This case tests that adding state 3 grows the state_list
    wf, z_out = tb.update_basis(ht, z_mem, [], [3])
    # Analytical: state_list should grow by 1
    assert 3 in tb.system.state_list
    assert len(tb.system.state_list) == len(state_list_before) + 1
    # Analytical: z_mem passes through unchanged
    assert z_out is z_mem


# ------------------------------------------------------------
# TEST: update_basis removing a state shrinks state_list
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_update_basis_remove_shrinks_state_list():
    '''update_basis with list_states_old shrinks system.state_list.'''
    # Setup: full 4-state basis, remove state 3
    tb = _make_initialized_tensor_basis()
    ht = _make_tensor_for_tb(tb)
    tb.eom = _MockEOM()
    z_mem = np.zeros(len(tb.mode.list_modeidx_abs), dtype=np.complex128)
    state_list_before = sorted(tb.system.state_list)
    # This case tests that removing state 3 shrinks the state_list
    wf, z_out = tb.update_basis(ht, z_mem, [3], [])
    # Analytical: state_list should shrink by 1
    assert 3 not in tb.system.state_list
    assert len(tb.system.state_list) == len(state_list_before) - 1
    # Analytical: wavefunction is returned (same object, mutated in place)
    assert wf is ht


# ------------------------------------------------------------
# TEST: update_basis roundtrip (add then remove) restores state_list
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_update_basis_roundtrip():
    '''Adding then removing a state returns state_list to original.'''
    # Setup: 3-state basis [0, 1, 2]
    sl_initial = [0, 1, 2]
    psi_3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
    tb = _make_initialized_tensor_basis(sl=sl_initial, psi=psi_3)
    ht = _make_tensor_for_tb(tb)
    tb.eom = _MockEOM()
    z_mem = np.zeros(len(tb.mode.list_modeidx_abs), dtype=np.complex128)
    state_list_original = sorted(tb.system.state_list)
    # This case tests that add followed by remove restores original state_list
    # Step 1: add state 3
    wf, z_out = tb.update_basis(ht, z_mem, [], [3])
    assert 3 in tb.system.state_list
    # Step 2: remove state 3
    wf, z_out = tb.update_basis(wf, z_out, [3], [])
    # Analytical: state_list should match original
    assert sorted(tb.system.state_list) == state_list_original
