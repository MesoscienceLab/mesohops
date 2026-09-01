import numpy as np
import pytest
import scipy as sp

from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_tensor_trajectory import HopsTensorTrajectory
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS

__title__ = 'Test of tensor low-temperature correction'
__author__ = 'A. Hartzell'
__maintainer__ = 'A. Hartzell'

# ============================================================
# System Parameters (duplicated from test_LTC_eom.py)
# ============================================================

noise_param = {
    'SEED': 0,
    'MODEL': 'FFT_FILTER',
    'TLEN': 50.0,
    'TAU': 1.0,
}

# --- 3-site system ---

T3_loperator = np.zeros([3, 3, 3], dtype=np.float64)
T3_loperator[0, 0, 0] = 1.0
T3_loperator[1, 1, 1] = 1.0
T3_loperator[2, 2, 2] = 1.0

sys_param_ltc = {
    'HAMILTONIAN': np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                            dtype=np.float64),
    'GW_SYSBATH': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                   [10.0, 10.0], [5.0, 5.0]],
    'L_HIER': [T3_loperator[0], T3_loperator[0], T3_loperator[1],
               T3_loperator[1], T3_loperator[2], T3_loperator[2]],
    'L_NOISE1': [T3_loperator[0], T3_loperator[0], T3_loperator[1],
                 T3_loperator[1], T3_loperator[2], T3_loperator[2],
                 T3_loperator[0], T3_loperator[1], T3_loperator[2]],
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [10.0, 10.0], [5.0, 5.0], [250.0, 1000.0],
                     [250.0, 2000.0], [250.0, 3000.0]],
    'PARAM_LT_CORR': [250.0 / 1000.0, 250.0 / 2000.0, 250.0 / 3000.0],
    'L_LT_CORR': [T3_loperator[0], T3_loperator[1], T3_loperator[2]],
}

sys_param_no_ltc = {
    'HAMILTONIAN': np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                            dtype=np.float64),
    'GW_SYSBATH': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                   [10.0, 10.0], [5.0, 5.0]],
    'L_HIER': [T3_loperator[0], T3_loperator[0], T3_loperator[1],
               T3_loperator[1], T3_loperator[2], T3_loperator[2]],
    'L_NOISE1': [T3_loperator[0], T3_loperator[0], T3_loperator[1],
                 T3_loperator[1], T3_loperator[2], T3_loperator[2],
                 T3_loperator[0], T3_loperator[1], T3_loperator[2]],
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [10.0, 10.0], [5.0, 5.0], [250.0, 1000.0],
                     [250.0, 2000.0], [250.0, 3000.0]],
}

# --- 4-site system (complex LTC coefficients, multi-state L-operators) ---

T3_loperator_4site = np.zeros([4, 4, 4], dtype=np.float64)
T3_loperator_4site[0, 0, 0] = 1.0
T3_loperator_4site[0, 1, 1] = 1.0
T3_loperator_4site[1, 0, 0] = 1.0
T3_loperator_4site[1, 2, 2] = 1.0
T3_loperator_4site[2, 2, 2] = 1.0
T3_loperator_4site[2, 3, 3] = 1.0
T3_loperator_4site[3, 1, 1] = 1.0
T3_loperator_4site[3, 3, 3] = 1.0

sys_param_ltc_4site = {
    'HAMILTONIAN': np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=np.float64),
    'GW_SYSBATH': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                   [10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    'L_HIER': [T3_loperator_4site[0], T3_loperator_4site[0],
               T3_loperator_4site[1], T3_loperator_4site[1],
               T3_loperator_4site[2], T3_loperator_4site[2],
               T3_loperator_4site[3], T3_loperator_4site[3]],
    'L_NOISE1': [T3_loperator_4site[0], T3_loperator_4site[0],
                 T3_loperator_4site[1], T3_loperator_4site[1],
                 T3_loperator_4site[2], T3_loperator_4site[2],
                 T3_loperator_4site[3], T3_loperator_4site[3],
                 T3_loperator_4site[0], T3_loperator_4site[1],
                 T3_loperator_4site[2], T3_loperator_4site[3]],
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [(250.0 + 10j), 1000.0], [(250.0 + 10j), 2000.0],
                     [(250.0 + 10j), 3000.0], [(250.0 + 10j), 4000.0]],
    'PARAM_LT_CORR': [(250.0 + 10j) / 1000.0, (250.0 + 10j) / 2000.0,
                      (250.0 + 10j) / 3000.0, (250.0 + 10j) / 4000.0],
    'L_LT_CORR': [T3_loperator_4site[0], T3_loperator_4site[1],
                  T3_loperator_4site[2], T3_loperator_4site[3]],
}

sys_param_no_ltc_4site = {
    'HAMILTONIAN': np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=np.float64),
    'GW_SYSBATH': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                   [10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    'L_HIER': [T3_loperator_4site[0], T3_loperator_4site[0],
               T3_loperator_4site[1], T3_loperator_4site[1],
               T3_loperator_4site[2], T3_loperator_4site[2],
               T3_loperator_4site[3], T3_loperator_4site[3]],
    'L_NOISE1': [T3_loperator_4site[0], T3_loperator_4site[0],
                 T3_loperator_4site[1], T3_loperator_4site[1],
                 T3_loperator_4site[2], T3_loperator_4site[2],
                 T3_loperator_4site[3], T3_loperator_4site[3],
                 T3_loperator_4site[0], T3_loperator_4site[1],
                 T3_loperator_4site[2], T3_loperator_4site[3]],
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [(250.0 + 10j), 1000.0], [(250.0 + 10j), 2000.0],
                     [(250.0 + 10j), 3000.0], [(250.0 + 10j), 4000.0]],
}

# --- Shared parameters ---

hier_param = {'MAXHIER': 1}
integrator_param = {
    'INTEGRATOR': 'RUNGE_KUTTA',
    'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
    'EARLY_INTEGRATOR_STEPS': 5,
    'INCHWORM_CAP': 5,
    'STATIC_BASIS': None,
}
tensor_param = {
    'MPS_EPSILON': 1e-12,
    'METHOD': 'fullstate',
    'BOND_DIM_MAX': 20,
}

t_max = 20.0
t_step = 2.0
H1_psi_0_3site = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
H1_psi_0_4site = np.array([0.7071, 0.5773, 0.4082, 0.1872],
                          dtype=np.complex128)
H1_psi_0_4site = H1_psi_0_4site / np.linalg.norm(H1_psi_0_4site)


# ============================================================
# Helpers
# ============================================================

def _run_tensor(sys_param, eom_param, psi_0, adaptive=False):
    """Run tensor HOPS and return psi trajectory as array."""
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    if adaptive:
        traj.make_adaptive(1e-5, 1e-5)
    traj.initialize(psi_0)
    traj.propagate(t_max, t_step)
    return np.array(traj.storage.data['psi_traj'])


def _run_vector(sys_param, eom_param, psi_0, adaptive=False):
    """Run vector HOPS and return psi trajectory as array."""
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    if adaptive:
        hops.make_adaptive(1e-5, 1e-5)
    hops.initialize(psi_0)
    hops.propagate(t_max, t_step)
    return np.array(hops.storage.data['psi_traj'])


# ============================================================
# TEST SUITE: LTC has effect on tensor dynamics (Group A)
# ============================================================


# ------------------------------------------------------------
# TEST: LTC modifies tensor dynamics — LINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_LTC_tensor_has_effect_linear():
    # This case tests that the low-temperature correction produces a
    # measurable change in the tensor psi trajectory under the LINEAR
    # equation of motion. Propagates with and without LTC on the same
    # 3-site system and asserts the trajectories diverge.
    eom_param = {
        'TIME_DEPENDENCE': False,
        'EQUATION_OF_MOTION': 'LINEAR',
    }
    psi_ltc = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site)
    psi_no_ltc = _run_tensor(sys_param_no_ltc, eom_param, H1_psi_0_3site)

    assert not np.allclose(psi_ltc, psi_no_ltc, atol=1e-6), (
        'LTC had no measurable effect on tensor LINEAR dynamics'
    )


# ------------------------------------------------------------
# TEST: LTC modifies tensor dynamics — NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_LTC_tensor_has_effect_nonlinear():
    # This case tests that the low-temperature correction produces a
    # measurable change in the tensor psi trajectory under the NONLINEAR
    # equation of motion.
    eom_param = {'EQUATION_OF_MOTION': 'NONLINEAR'}
    psi_ltc = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site)
    psi_no_ltc = _run_tensor(sys_param_no_ltc, eom_param, H1_psi_0_3site)

    assert not np.allclose(psi_ltc, psi_no_ltc, atol=1e-6), (
        'LTC had no measurable effect on tensor NONLINEAR dynamics'
    )


# ------------------------------------------------------------
# TEST: LTC modifies tensor dynamics — NORMALIZED NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_LTC_tensor_has_effect_normalized_nonlinear():
    # This case tests that the low-temperature correction produces a
    # measurable change in the tensor psi trajectory under the
    # NORMALIZED NONLINEAR equation of motion.
    eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    psi_ltc = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site)
    psi_no_ltc = _run_tensor(sys_param_no_ltc, eom_param, H1_psi_0_3site)

    assert not np.allclose(psi_ltc, psi_no_ltc, atol=1e-6), (
        'LTC had no measurable effect on tensor NORMALIZED NONLINEAR dynamics'
    )


# ------------------------------------------------------------
# TEST: LTC modifies tensor dynamics — adaptive, 3-site
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.xfail(reason='tensor adaptive basis not yet working')
def test_LTC_tensor_has_effect_adaptive():
    # This case tests that the low-temperature correction produces a
    # measurable change in the tensor psi trajectory with adaptive
    # basis on the 3-site system.
    eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    psi_ltc = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site,
                          adaptive=True)
    psi_no_ltc = _run_tensor(sys_param_no_ltc, eom_param, H1_psi_0_3site,
                             adaptive=True)

    assert not np.allclose(psi_ltc, psi_no_ltc, atol=1e-6), (
        'LTC had no measurable effect on adaptive tensor dynamics'
    )


# ------------------------------------------------------------
# TEST: LTC modifies tensor dynamics — adaptive, 4-site multiparticle
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.xfail(reason='tensor adaptive basis not yet working')
def test_LTC_tensor_has_effect_adaptive_multiparticle():
    # This case tests that the low-temperature correction produces a
    # measurable change in the tensor psi trajectory with adaptive
    # basis on the 4-site system with complex LTC coefficients and
    # multi-state L-operators.
    eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    psi_ltc = _run_tensor(sys_param_ltc_4site, eom_param, H1_psi_0_4site,
                          adaptive=True)
    psi_no_ltc = _run_tensor(sys_param_no_ltc_4site, eom_param,
                             H1_psi_0_4site, adaptive=True)

    assert not np.allclose(psi_ltc, psi_no_ltc, atol=1e-6), (
        'LTC had no measurable effect on adaptive multiparticle tensor dynamics'
    )


# ============================================================
# TEST SUITE: Tensor LTC matches vector LTC (Group B)
# ============================================================


# ------------------------------------------------------------
# TEST: Tensor LTC matches vector — LINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_LTC_tensor_vs_vector_linear():
    # This case tests that tensor HOPS with LTC produces the same psi
    # trajectory as vector HOPS with LTC under the LINEAR equation of
    # motion. Uses the 3-site system with real LTC coefficients.
    eom_param = {
        'TIME_DEPENDENCE': False,
        'EQUATION_OF_MOTION': 'LINEAR',
    }
    psi_tensor = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site)
    psi_vector = _run_vector(sys_param_ltc, eom_param, H1_psi_0_3site)

    n_steps = min(len(psi_tensor), len(psi_vector))
    max_err = max(
        np.linalg.norm(psi_tensor[i] - psi_vector[i])
        for i in range(n_steps)
    )
    # Relaxed to 2e-9: this 3-site system with 9 noise modes has a
    # baseline tensor-vs-vector discrepancy of ~1.7e-9 even without LTC,
    # caused by accumulated SVD truncation (epsilon=1e-12) over 10 steps.
    assert max_err < 2e-9, (
        f'Tensor LTC diverged from vector LTC (LINEAR): '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Tensor LTC matches vector — NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_LTC_tensor_vs_vector_nonlinear():
    # This case tests that tensor HOPS with LTC produces the same psi
    # trajectory as vector HOPS with LTC under the NONLINEAR equation
    # of motion.
    eom_param = {'EQUATION_OF_MOTION': 'NONLINEAR'}
    psi_tensor = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site)
    psi_vector = _run_vector(sys_param_ltc, eom_param, H1_psi_0_3site)

    n_steps = min(len(psi_tensor), len(psi_vector))
    max_err = max(
        np.linalg.norm(psi_tensor[i] - psi_vector[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Tensor LTC diverged from vector LTC (NONLINEAR): '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Tensor LTC matches vector — NORMALIZED NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_LTC_tensor_vs_vector_normalized_nonlinear():
    # This case tests that tensor HOPS with LTC produces the same psi
    # trajectory as vector HOPS with LTC under the NORMALIZED NONLINEAR
    # equation of motion.
    eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    psi_tensor = _run_tensor(sys_param_ltc, eom_param, H1_psi_0_3site)
    psi_vector = _run_vector(sys_param_ltc, eom_param, H1_psi_0_3site)

    n_steps = min(len(psi_tensor), len(psi_vector))
    max_err = max(
        np.linalg.norm(psi_tensor[i] - psi_vector[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Tensor LTC diverged from vector LTC (NORMALIZED NONLINEAR): '
        f'max wf error = {max_err:.2e}'
    )
