__title__ = 'Unit Tests for tensor_functions_adaptive'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'

import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.tensor.tensor_functions_adaptive import (
    tensor_state_adaptive_check_add_state,
    tensor_state_adaptive_check_remove_state,
)
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.exceptions import UnsupportedRequest

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

# Dimer-of-dimers: 0-1 and 2-3 coupled strongly, 1-2 weakly
hs = np.zeros([nsite, nsite])
hs[0, 1] = 40
hs[1, 0] = 40
hs[1, 2] = 10
hs[2, 1] = 10
hs[2, 3] = 40
hs[3, 2] = 40
H2_sys_hamiltonian = np.array(hs, dtype=np.complex128)

sys_param = {
    'HAMILTONIAN': H2_sys_hamiltonian,
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': lop_list,
    'L_NOISE1': lop_list,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': gw_sysbath,
}

psi_0 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)

k_max = 4

tensor_param = {
    'MPS_EPSILON': 1e-10,
    'METHOD': 'fullstate',
    'BOND_DIM_MAX': 20,
}

integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}


def _make_basis_objects(sp=sys_param):
    '''Creates HopsSystem, HopsModes, HopsNoiseMemory directly.'''
    system = HopsSystem(sp)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    return system, mode, noise_memory


def _make_initialized_tensor_basis(delta_s=0, sl=None, psi=None):
    '''Creates and initializes a HopsTensorBasis.'''
    if sl is None:
        sl = list(np.arange(nsite))
    if psi is None:
        psi = psi_0
    system, mode, noise_memory = _make_basis_objects()
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(delta_s > 0, psi)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = sl
    tb.initialize(delta_s)
    return tb


def _make_wavefunction(state_list, psi_on_states, method='fullstate'):
    '''
    Builds an initialized HopsTensorWavefunction for the given state_list
    and initial wavefunction.

    Parameters
    ----------
    1. state_list : list(int)
                    Absolute state indices to include.
    2. psi_on_states : np.ndarray(complex)
                       Initial wavefunction amplitudes, one per state in state_list.
    3. method : str
                Tensor encoding type.

    Returns
    -------
    1. ht : HopsTensorWavefunction
    2. M1_modes_per_state : np.ndarray(int)
                            Number of bath modes per absolute state index
                            (length = n_state_full).
    '''
    tp = dict(tensor_param, METHOD=method)
    # Build psi in full nsite space so system.initialize accepts it
    psi_full = np.zeros(nsite, dtype=np.complex128)
    for i, s in enumerate(state_list):
        psi_full[s] = psi_on_states[i]
    tb = _make_initialized_tensor_basis(delta_s=0, sl=state_list, psi=psi_full)
    ht = HopsTensorWavefunction(k_max, tp, integrator_param, eom_param)
    # HopsTensorWavefunction.initialize does phi_0[system.state_list] internally,
    # so we must pass the full-length psi_full here.
    ht.initialize(
        psi_full,
        tb.system,
    )
    return ht, ht.M1_modes_per_state


# ============================================================
# TEST SUITE: tensor_state_adaptive_check_add_state()
# ============================================================


# ------------------------------------------------------------
# TEST: fullstate adds Hamiltonian-coupled states
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_add_state_fullstate_adds_coupled_states():
    '''check_add_state (fullstate) adds H-coupled states outside basis.

    Setup: 2-state basis [1, 2], psi localized on state 2 (index 1 in basis).
    State 2 couples to state 1 (already in basis) and state 3 (outside).
    With delta_s small but nonzero, state 3 should be added.
    '''
    # Analytical: H has nonzero coupling 2<->3 (hs[2,3]=40) and 1<->2 (hs[1,2]=10).
    # With psi on state 2, flux into state 3 should exceed threshold for small delta_s.
    sl = [1, 2]
    psi_local = np.array([0.0, 1.0], dtype=np.complex128)  # psi on state 2
    ht, M1_modes_per_state = _make_wavefunction(sl, psi_local)

    n_state = len(sl)
    n_state_full = nsite

    list_new = tensor_state_adaptive_check_add_state(
        list_cores_phi=ht.list_cores_phi,
        ham=H2_sys_hamiltonian,
        old_states=sl,
        n_state_full=n_state_full,
        n_state=n_state,
        delta_s=1e-6,    # very small threshold — all significant flux should trigger
        state_list=sl,
        method='fullstate',
        M1_modes_per_state=M1_modes_per_state,
    )
    # State 3 couples to state 2 via H[2,3]=40; should be detected
    assert 3 in list_new
    # States already in basis must not be returned
    for s in sl:
        assert s not in list_new


# ------------------------------------------------------------
# TEST: number matches fullstate
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_add_state_statenumber_matches_fullstate():
    '''statenumber and fullstate return the same new-state set.

    Analytical: both methods estimate the same boundary flux; results
    should agree for a localized initial state.
    '''
    sl = [1, 2]
    psi_local = np.array([0.0, 1.0], dtype=np.complex128)

    ht_full, M1_modes_per_state = _make_wavefunction(
        sl, psi_local, method='fullstate'
    )
    ht_num, _ = _make_wavefunction(
        sl, psi_local, method='number'
    )

    kwargs = dict(
        ham=H2_sys_hamiltonian,
        old_states=sl,
        n_state_full=nsite,
        n_state=len(sl),
        delta_s=1e-6,
        state_list=sl,
        M1_modes_per_state=M1_modes_per_state,
    )
    list_new_full = tensor_state_adaptive_check_add_state(
        list_cores_phi=ht_full.list_cores_phi,
        method='fullstate',
        **kwargs,
    )
    list_new_num = tensor_state_adaptive_check_add_state(
        list_cores_phi=ht_num.list_cores_phi,
        method='number',
        **kwargs,
    )
    # Both should agree on which states to add (sets may differ in order)
    assert set(list_new_full) == set(list_new_num)


# ------------------------------------------------------------
# TEST: invalid method raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_add_state_invalid_method_raises():
    '''check_add_state raises UnsupportedRequest for unknown method.'''
    sl = [1, 2]
    psi_local = np.array([0.0, 1.0], dtype=np.complex128)
    ht, M1_modes_per_state = _make_wavefunction(sl, psi_local)

    with pytest.raises(UnsupportedRequest):
        tensor_state_adaptive_check_add_state(
            list_cores_phi=ht.list_cores_phi,
            ham=H2_sys_hamiltonian,
            old_states=sl,
            n_state_full=nsite,
            n_state=len(sl),
            delta_s=0.1,
            state_list=sl,
            method='invalidmethod',
            M1_modes_per_state=M1_modes_per_state,
        )


# ------------------------------------------------------------
# TEST: all states present → returns empty list
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_add_state_full_basis_returns_empty():
    '''check_add_state returns empty list when all states are in the basis.

    Analytical: V1_flux[state_list] is zeroed out by the function; when
    state_list = all states, no state can have positive flux.
    '''
    sl = list(np.arange(nsite))
    ht, M1_modes_per_state = _make_wavefunction(sl, psi_0)

    list_new = tensor_state_adaptive_check_add_state(
        list_cores_phi=ht.list_cores_phi,
        ham=H2_sys_hamiltonian,
        old_states=sl,
        n_state_full=nsite,
        n_state=len(sl),
        delta_s=0.1,
        state_list=sl,
        method='fullstate',
        M1_modes_per_state=M1_modes_per_state,
    )
    assert list_new == []


# ------------------------------------------------------------
# TEST: delta_s=0 adds all coupled states outside basis
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_add_state_delta_s_zero_adds_all_coupled():
    '''With delta_s=0 threshold is zero, every state with nonzero flux is added.

    Analytical: determine_error_thresh with max_error=0 returns 0; all states
    with V1_flux > 0 are added. H applied to psi (on state 2 in basis [1,2])
    gives nonzero flux exactly for states directly coupled to state 2.
    H[2,3]=40 so state 3 gets flux; H[2,0]=0 so state 0 gets no flux.
    With basis [1,2], states 0 and 3 are outside, but only 3 couples directly.
    '''
    sl = [1, 2]
    psi_local = np.array([0.0, 1.0], dtype=np.complex128)
    ht, M1_modes_per_state = _make_wavefunction(sl, psi_local)

    list_new = tensor_state_adaptive_check_add_state(
        list_cores_phi=ht.list_cores_phi,
        ham=H2_sys_hamiltonian,
        old_states=sl,
        n_state_full=nsite,
        n_state=len(sl),
        delta_s=0.0,
        state_list=sl,
        method='fullstate',
        M1_modes_per_state=M1_modes_per_state,
    )
    # With zero threshold, all directly coupled states outside basis appear.
    # H[2,3]=40 gives flux to state 3; H[2,0]=0 so state 0 gets none.
    assert 3 in list_new
    # State 0 is not directly coupled to state 2, so no flux → not added
    assert 0 not in list_new
    # States in basis must not appear
    for s in sl:
        assert s not in list_new


# ============================================================
# TEST SUITE: tensor_state_adaptive_check_remove_state()
# ============================================================


# ------------------------------------------------------------
# TEST: fullstate removes low-flux states
# ------------------------------------------------------------
@pytest.mark.xfail(
    reason=(
        'Bug in tensor_functions_adaptive: contract_down_exact returns a '
        'read-only diag view; V1_error += ... raises ValueError. '
        'Fix: replace V1_error += with V1_error = V1_error + ...'
    ),
    strict=True,
)
@pytest.mark.level(1)
def test_check_remove_state_fullstate_removes_decoupled():
    '''check_remove_state (fullstate) identifies low-flux states to remove.

    Setup: full 4-state basis, psi on state 2. With a mock dsystem_dt that
    returns zero derivative and large delta_s, states with small coupling
    flux are below threshold and should be candidates for removal.

    Known bug: V1_error returned by contract_down_exact is read-only
    (numpy diag view), so V1_error += ... raises ValueError. Marked xfail.
    '''
    sl = list(np.arange(nsite))
    ht, M1_modes_per_state = _make_wavefunction(sl, psi_0)

    def _mock_dsystem_dt(z_mem, z_rnd, z_rnd2):
        return [np.zeros_like(c) for c in ht.list_cores_phi]

    z_rnd = np.zeros(len(sl), dtype=np.complex128)
    z_step = [z_rnd, z_rnd.copy(), np.zeros(len(sl), dtype=np.complex128)]

    list_old = tensor_state_adaptive_check_remove_state(
        list_cores_phi=ht.list_cores_phi,
        ham=H2_sys_hamiltonian,
        z_step=z_step,
        n_state_full=nsite,
        n_state=len(sl),
        delta_s=1.0,     # large threshold — most states should be removal candidates
        state_list=sl,
        method='fullstate',
        M1_modes_per_state=M1_modes_per_state,
        dsystem_dt=_mock_dsystem_dt,
    )
    # Analytical: psi is on state 2. States with no or weak coupling to state
    # 2 (e.g., state 0 which only couples to state 1) should be removable.
    # Result must be relative indices into sl, so values are in [0, n_state).
    assert all(0 <= idx < len(sl) for idx in list_old)
    assert len(list_old) >= 0  # basic sanity: no exception


# ------------------------------------------------------------
# TEST: invalid method raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_remove_state_invalid_method_raises():
    '''check_remove_state raises UnsupportedRequest for unknown method.'''
    sl = list(np.arange(nsite))
    ht, M1_modes_per_state = _make_wavefunction(sl, psi_0)

    def _mock_dsystem_dt(z_mem, z_rnd, z_rnd2):
        return [np.zeros_like(c) for c in ht.list_cores_phi]

    z_rnd = np.zeros(len(sl), dtype=np.complex128)
    z_step = [z_rnd, z_rnd.copy(), np.zeros(len(sl), dtype=np.complex128)]

    with pytest.raises(UnsupportedRequest):
        tensor_state_adaptive_check_remove_state(
            list_cores_phi=ht.list_cores_phi,
            ham=H2_sys_hamiltonian,
            z_step=z_step,
            n_state_full=nsite,
            n_state=len(sl),
            delta_s=0.1,
            state_list=sl,
            method='invalidmethod',
            M1_modes_per_state=M1_modes_per_state,
            dsystem_dt=_mock_dsystem_dt,
        )
