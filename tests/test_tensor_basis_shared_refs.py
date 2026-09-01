import numpy as np
import pytest
import scipy as sp

from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_tensor_trajectory import HopsTensorTrajectory
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp

__title__ = 'Integration Tests for Tensor Basis Shared References'
__author__ = 'A. Hartzell'
__maintainer__ = 'A. Hartzell'

# ============================================================
# Shared Setup
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

sys_param = {
    'HAMILTONIAN': np.array(np.zeros([nsite, nsite]), dtype=np.complex128),
    'GW_SYSBATH': list_gw_sysbath,
    'L_HIER': list_lop,
    'L_NOISE1': list_lop,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': list_gw_sysbath,
}
sys_param['HAMILTONIAN'][0, 1] = 40
sys_param['HAMILTONIAN'][1, 0] = 40
sys_param['HAMILTONIAN'][1, 2] = 10
sys_param['HAMILTONIAN'][2, 1] = 10
sys_param['HAMILTONIAN'][2, 3] = 40
sys_param['HAMILTONIAN'][3, 2] = 40

noise_param = {
    'SEED': 0,
    'MODEL': 'FFT_FILTER',
    'TLEN': 25000.0,
    'TAU': 1.0,
}

hier_param = {'MAXHIER': 2}
eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
tensor_param = {
    'MPS_EPSILON': 1e-10,
    'METHOD': 'fullstate',
    'BOND_DIM_MAX': 20,
}

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0
psi_0 = psi_0 / np.linalg.norm(psi_0)


def _make_traj():
    return HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )


# ============================================================
# TEST SUITE: HopsTensorTrajectory() — shared reference identity
# ============================================================


# ------------------------------------------------------------
# TEST: tensor_basis.system is basis.system
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_system_is_shared():
    '''tensor_basis.system must be the same object as basis.system.'''
    # This case tests that tensor_basis holds a reference to the same
    # HopsSystem instance as the HOPS basis, not a copy.
    traj = _make_traj()
    assert traj.tensor_basis.system is traj.basis.system


# ------------------------------------------------------------
# TEST: tensor_basis.mode is basis.mode
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_mode_is_shared():
    '''tensor_basis.mode must be the same object as basis.mode.'''
    # This case tests that tensor_basis holds a reference to the same
    # HopsModes instance as the HOPS basis, not a copy.
    traj = _make_traj()
    assert traj.tensor_basis.mode is traj.basis.mode


# ------------------------------------------------------------
# TEST: tensor_basis.noise_memory is basis.noise_memory
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_noise_memory_is_shared():
    '''tensor_basis.noise_memory must be the same object as basis.noise_memory.'''
    # This case tests that tensor_basis holds a reference to the same
    # HopsNoiseMemory instance as the HOPS basis, not a copy.
    traj = _make_traj()
    assert traj.tensor_basis.noise_memory is traj.basis.noise_memory


# ------------------------------------------------------------
# TEST: State list mutation through tensor_basis is visible through basis
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_state_list_mutation_visibility():
    # This case tests that mutating state_list through the tensor_basis path
    # is immediately visible through the basis path, confirming the two paths
    # reference the same underlying HopsSystem object.
    traj = _make_traj()
    # Mutate state_list through tensor_basis path
    traj.tensor_basis.system.state_list = [0, 1]
    # Invariant: visible through basis path (same object)
    np.testing.assert_array_equal(traj.basis.system.state_list, [0, 1])


