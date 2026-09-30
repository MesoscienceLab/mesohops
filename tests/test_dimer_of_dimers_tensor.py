import numpy as np
import pytest
import scipy as sp

from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_tensor_trajectory import HopsTensorTrajectory
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp

__title__ = 'Dimer of Dimers: Tensor HOPS vs Vector HOPS'
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

eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
hier_param = {'MAXHIER': 2, 'TRUNCATION_METHOD': 'rectangular'}

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0
psi_0 = psi_0 / np.linalg.norm(psi_0)

t_max = 200.0
t_step = 4.0


# ============================================================
# Helpers
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
    return np.array(hops.storage['psi_traj'])


def _run_tensor_hops(method, bond_dim_max=20, mps_epsilon=1e-10,
                     flag_mpo_optimize=True):
    """Runs tensor HOPS and returns psi_traj as array."""
    tensor_param = {
        'MPS_EPSILON': mps_epsilon,
        'METHOD': method,
        'BOND_DIM_MAX': bond_dim_max,
        'FLAG_MPO_OPTIMIZE': flag_mpo_optimize,
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
# Shared Fixtures
# ============================================================
# Each propagation runs once per test session and is reused across tests.

@pytest.fixture(scope='module')
def psi_vector():
    """Vector HOPS trajectory (computed once per module)."""
    return _run_vector_hops()


@pytest.fixture(scope='module')
def psi_fullstate():
    """Fullstate tensor HOPS trajectory (computed once per module)."""
    return _run_tensor_hops(method='fullstate')


@pytest.fixture(
    scope='module', params=[True, False], ids=['optimized', 'general'],
)
def psi_statenumber_nn(request):
    """Statenumber NN tensor HOPS trajectory, once per FLAG_MPO_OPTIMIZE
    setting: the chain combined generator and the general two-MPO path.
    """
    return _run_tensor_hops(
        method='number', flag_mpo_optimize=request.param,
    )


# ============================================================
# TEST SUITE: propagate() — vector HOPS vs tensor HOPS agreement
# ============================================================
# These tests verify that the tensor HOPS EOM produces the same
# physical wavefunction trajectory as vector HOPS on the dimer-of-dimers
# system, with tight SVD convergence (epsilon=1e-10, bond_dim_max=20).
# At these parameters the hierarchy is small enough that SVD compression
# is essentially lossless, giving tensor-vs-vector agreement to ~1e-10.
# Tolerance set to 1e-9 to catch coupling-prefactor or MPO-wiring bugs
# that would accumulate over 50 RK4 steps.

# ------------------------------------------------------------
# TEST: Fullstate representation matches vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fullstate_matches_vector_hops(psi_vector, psi_fullstate):
    # This case tests that the fullstate tensor HOPS
    # produces the same physical wavefunction trajectory as standard
    # vector HOPS on the dimer-of-dimers system.
    assert len(psi_vector) == len(psi_fullstate), "Trajectory lengths differ"
    n_steps = len(psi_vector)

    # SVD compression errors accumulate over time steps, so we check the
    # maximum wavefunction error across the full trajectory.
    max_err = max(
        np.linalg.norm(psi_fullstate[i] - psi_vector[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Fullstate tensor HOPS diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Statenumber NN representation matches vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_statenumber_nn_matches_vector_hops(psi_vector, psi_statenumber_nn):
    # This case tests that the number with
    # nearest-neighbor Hamiltonian MPO produces the same dynamics
    # as standard vector HOPS.
    assert len(psi_vector) == len(psi_statenumber_nn), "Trajectory lengths differ"
    n_steps = len(psi_vector)

    max_err = max(
        np.linalg.norm(psi_statenumber_nn[i] - psi_vector[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Statenumber NN tensor HOPS diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: propagate() — non-nearest-neighbor Hamiltonian
# ============================================================
# Long-range couplings H[0,2]=5, H[0,3]=3 break the NN MPO path and
# exercise the general (statenumber) MPO builder end-to-end.

H2_ham_nonnn = np.zeros([nsite, nsite])
H2_ham_nonnn[0, 1] = 40
H2_ham_nonnn[1, 0] = 40
H2_ham_nonnn[1, 2] = 10
H2_ham_nonnn[2, 1] = 10
H2_ham_nonnn[2, 3] = 40
H2_ham_nonnn[3, 2] = 40
H2_ham_nonnn[0, 2] = 5
H2_ham_nonnn[2, 0] = 5
H2_ham_nonnn[0, 3] = 3
H2_ham_nonnn[3, 0] = 3

_sys_param_nonnn = dict(sys_param)
_sys_param_nonnn['HAMILTONIAN'] = np.array(H2_ham_nonnn, dtype=np.complex128)


def _run_vector_hops_nonnn():
    """Runs standard vector HOPS with non-NN Hamiltonian."""
    hops = HOPS(
        _sys_param_nonnn,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)
    hops.propagate(t_max, t_step)
    return np.array(hops.storage['psi_traj'])


def _run_tensor_hops_nonnn(method, bond_dim_max=30, mps_epsilon=1e-12):
    """Runs tensor HOPS with non-NN Hamiltonian.

    The MPS budget is tighter than BOND_DIM_MAX=20 / MPS_EPSILON=1e-10 because
    at that budget the gate measures MPS truncation rather than the MPO: the
    number method sits 1.4e-9 from vector HOPS and the fullstate 2.3e-10,
    against a 1e-9 threshold, and which side of it they fall on depends on the
    MPO's bond profile rather than on the operator it encodes. At these
    settings every comparison in the non-nearest-neighbor group is 1e-11 or
    below.
    """
    tensor_param = {
        'MPS_EPSILON': mps_epsilon,
        'METHOD': method,
        'BOND_DIM_MAX': bond_dim_max,
    }
    traj = HopsTensorTrajectory(
        system_param=_sys_param_nonnn,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    traj.propagate(t_max, t_step)
    return np.array(traj.storage['psi_traj'])


@pytest.fixture(scope='module')
def psi_vector_nonnn():
    """Vector HOPS trajectory with non-NN Hamiltonian (computed once per module)."""
    return _run_vector_hops_nonnn()


@pytest.fixture(scope='module')
def psi_fullstate_nonnn():
    """Fullstate tensor HOPS trajectory with non-NN Hamiltonian."""
    return _run_tensor_hops_nonnn(method='fullstate')


@pytest.fixture(scope='module')
def psi_statenumber_general_nonnn():
    """Statenumber general tensor HOPS trajectory with non-NN Hamiltonian."""
    return _run_tensor_hops_nonnn(method='number')


# ------------------------------------------------------------
# TEST: Non-NN fullstate matches vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fullstate_nonnn_matches_vector_hops(
    psi_vector_nonnn, psi_fullstate_nonnn,
):
    # This case tests that fullstate tensor HOPS handles long-range
    # couplings correctly (fullstate embeds the Hamiltonian into the
    # first MPO core, so non-NN structure is handled implicitly).
    assert len(psi_vector_nonnn) == len(psi_fullstate_nonnn), "Trajectory lengths differ"
    n_steps = len(psi_vector_nonnn)

    max_err = max(
        np.linalg.norm(psi_fullstate_nonnn[i] - psi_vector_nonnn[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Fullstate tensor HOPS (non-NN Ham) diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Non-NN statenumber general representation matches vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_statenumber_general_nonnn_matches_vector_hops(
    psi_vector_nonnn, psi_statenumber_general_nonnn,
):
    # This case tests that the statenumber general MPO handles a non-nearest-
    # neighbor Hamiltonian correctly. Long-range couplings (H[0,2]=5, H[0,3]=3)
    # require the general MPO builder path; this verifies the result still
    # matches standard vector HOPS to confirm correctness.
    assert len(psi_vector_nonnn) == len(psi_statenumber_general_nonnn), "Trajectory lengths differ"
    n_steps = len(psi_vector_nonnn)

    max_err = max(
        np.linalg.norm(psi_statenumber_general_nonnn[i] - psi_vector_nonnn[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Statenumber general tensor HOPS (non-NN Ham) diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Non-NN fullstate vs statenumber agree
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fullstate_vs_statenumber_nonnn(
    psi_fullstate_nonnn, psi_statenumber_general_nonnn,
):
    # This case tests that both representations agree for the non-NN
    # Hamiltonian. The fullstate MPO embeds the full Hamiltonian in
    # one core while the statenumber general MPO uses daisy-chained
    # transfer matrices — both must produce the same result.
    assert len(psi_fullstate_nonnn) == len(psi_statenumber_general_nonnn), "Trajectory lengths differ"
    n_steps = len(psi_fullstate_nonnn)

    max_err = max(
        np.linalg.norm(psi_fullstate_nonnn[i] - psi_statenumber_general_nonnn[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Fullstate vs statenumber disagree for non-NN Ham: '
        f'max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: propagate() — linear EOM
# ============================================================
# For linear HOPS the wavefunction is not renormalized, so its norm decays
# over time. The tensor and vector trajectories must agree element-wise on
# the raw (unnormalized) wavefunction at each stored timestep.

eom_param_linear = {'EQUATION_OF_MOTION': 'LINEAR'}


def _run_vector_linear():
    """Runs vector HOPS with linear EOM and returns psi_traj."""
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param_linear,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)
    hops.propagate(t_max, t_step)
    return np.array(hops.storage['psi_traj'])


def _run_tensor_linear(method, bond_dim_max=20, mps_epsilon=1e-10):
    """Runs tensor HOPS with linear EOM and returns psi_traj."""
    tensor_param = {
        'MPS_EPSILON': mps_epsilon,
        'METHOD': method,
        'BOND_DIM_MAX': bond_dim_max,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param_linear,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    traj.propagate(t_max, t_step)
    return np.array(traj.storage['psi_traj'])


@pytest.fixture(scope='module')
def psi_vector_linear():
    """Vector linear HOPS trajectory (computed once per module)."""
    return _run_vector_linear()


@pytest.fixture(scope='module')
def psi_fullstate_linear():
    """Fullstate tensor linear HOPS trajectory (computed once per module)."""
    return _run_tensor_linear(method='fullstate')


@pytest.fixture(scope='module')
def psi_statenumber_linear():
    """Statenumber tensor linear HOPS trajectory (computed once per module)."""
    return _run_tensor_linear(method='number')


# ------------------------------------------------------------
# TEST: Fullstate linear tensor HOPS matches vector linear HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_linear_fullstate_matches_vector(psi_vector_linear, psi_fullstate_linear):
    # This case tests that fullstate tensor HOPS with the
    # linear EOM produces the same unnormalized wavefunction trajectory
    # as vector HOPS. The norm is allowed to decay; what matters is
    # element-wise agreement of the raw psi_traj. The tighter tolerance
    # (vs 1e-6 for nonlinear) reflects the absence of nonlinear norm
    # correction errors.
    assert len(psi_vector_linear) == len(psi_fullstate_linear), "Trajectory lengths differ"
    n_steps = len(psi_vector_linear)

    max_err = max(
        np.linalg.norm(psi_fullstate_linear[i] - psi_vector_linear[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-10, (
        f'Fullstate linear tensor HOPS diverged from vector linear HOPS: '
        f'max wf error = {max_err:.2e}'
    )

    # This case tests that the unnormalized wavefunction norm decays,
    # confirming the linear EOM is not accidentally renormalizing.
    norm_first = np.linalg.norm(psi_fullstate_linear[0])
    norm_last = np.linalg.norm(psi_fullstate_linear[-1])
    assert norm_last < norm_first, (
        f'Linear EOM norm should decay: first={norm_first:.6f}, '
        f'last={norm_last:.6f}'
    )


# ------------------------------------------------------------
# TEST: Statenumber linear tensor HOPS matches vector linear HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_linear_statenumber_matches_vector(psi_vector_linear, psi_statenumber_linear):
    # This case tests that number tensor HOPS with the
    # linear EOM produces the same unnormalized wavefunction trajectory
    # as vector HOPS.
    assert len(psi_vector_linear) == len(psi_statenumber_linear), "Trajectory lengths differ"
    n_steps = len(psi_vector_linear)

    max_err = max(
        np.linalg.norm(psi_statenumber_linear[i] - psi_vector_linear[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-10, (
        f'Statenumber linear tensor HOPS diverged from vector linear HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: propagate() — NONLINEAR EOM (no norm correction)
# ============================================================
# NONLINEAR has <L> feedback but no norm correction. This is a distinct
# code path from both LINEAR (no feedback, no norm) and NORMALIZED
# NONLINEAR (feedback + norm). The norm decays because there is no
# renormalization.

eom_param_nonlinear = {'EQUATION_OF_MOTION': 'NONLINEAR'}


def _run_vector_nonlinear():
    """Runs vector HOPS with NONLINEAR EOM."""
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param_nonlinear,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)
    hops.propagate(t_max, t_step)
    return np.array(hops.storage['psi_traj'])


def _run_tensor_nonlinear(method, bond_dim_max=20, mps_epsilon=1e-10):
    """Runs tensor HOPS with NONLINEAR EOM."""
    tensor_param = {
        'MPS_EPSILON': mps_epsilon,
        'METHOD': method,
        'BOND_DIM_MAX': bond_dim_max,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param_nonlinear,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    traj.propagate(t_max, t_step)
    return np.array(traj.storage['psi_traj'])


@pytest.fixture(scope='module')
def psi_vector_nonlinear():
    """Vector NONLINEAR HOPS trajectory (computed once per module)."""
    return _run_vector_nonlinear()


@pytest.fixture(scope='module')
def psi_fullstate_nonlinear():
    """Fullstate tensor NONLINEAR HOPS trajectory (computed once per module)."""
    return _run_tensor_nonlinear(method='fullstate')


@pytest.fixture(scope='module')
def psi_statenumber_nonlinear():
    """Statenumber tensor NONLINEAR HOPS trajectory (computed once per module)."""
    return _run_tensor_nonlinear(method='number')


# ------------------------------------------------------------
# TEST: Fullstate NONLINEAR tensor matches vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_nonlinear_fullstate_matches_vector(
    psi_vector_nonlinear, psi_fullstate_nonlinear,
):
    # This case tests that fullstate tensor HOPS with the NONLINEAR EOM
    # (<L> feedback, no norm correction) matches vector HOPS.
    assert len(psi_vector_nonlinear) == len(psi_fullstate_nonlinear), "Trajectory lengths differ"
    n_steps = len(psi_vector_nonlinear)

    max_err = max(
        np.linalg.norm(psi_fullstate_nonlinear[i] - psi_vector_nonlinear[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Fullstate NONLINEAR tensor HOPS diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Statenumber NONLINEAR tensor matches vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_nonlinear_statenumber_matches_vector(
    psi_vector_nonlinear, psi_statenumber_nonlinear,
):
    # This case tests that statenumber tensor HOPS with the NONLINEAR EOM
    # matches vector HOPS.
    assert len(psi_vector_nonlinear) == len(psi_statenumber_nonlinear), "Trajectory lengths differ"
    n_steps = len(psi_vector_nonlinear)

    max_err = max(
        np.linalg.norm(psi_statenumber_nonlinear[i] - psi_vector_nonlinear[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Statenumber NONLINEAR tensor HOPS diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: propagate() — fullstate vs statenumber agreement
# ============================================================
# Cross-comparison between representations catches bugs that affect
# both equally (where both diverge from vector HOPS by the same amount).

# ------------------------------------------------------------
# TEST: Fullstate and statenumber NN agree for NORMALIZED NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fullstate_vs_statenumber_normalized_nonlinear(
    psi_fullstate, psi_statenumber_nn,
):
    # This case tests that both tensor representations produce identical
    # trajectories, independent of the vector HOPS reference.
    assert len(psi_fullstate) == len(psi_statenumber_nn), "Trajectory lengths differ"
    n_steps = len(psi_fullstate)

    max_err = max(
        np.linalg.norm(psi_fullstate[i] - psi_statenumber_nn[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Fullstate vs statenumber disagree for NORMALIZED NONLINEAR: '
        f'max wf error = {max_err:.2e}'
    )


# ------------------------------------------------------------
# TEST: Fullstate and statenumber agree for NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fullstate_vs_statenumber_nonlinear(
    psi_fullstate_nonlinear, psi_statenumber_nonlinear,
):
    # This case tests that both representations agree for the NONLINEAR
    # EOM (<L> feedback, no norm correction).
    assert len(psi_fullstate_nonlinear) == len(psi_statenumber_nonlinear), "Trajectory lengths differ"
    n_steps = len(psi_fullstate_nonlinear)

    max_err = max(
        np.linalg.norm(psi_fullstate_nonlinear[i] - psi_statenumber_nonlinear[i])
        for i in range(n_steps)
    )
    assert max_err < 1e-9, (
        f'Fullstate vs statenumber disagree for NONLINEAR: '
        f'max wf error = {max_err:.2e}'
    )


# ============================================================
# TEST SUITE: propagate() — ring and star Hamiltonians
# ============================================================
# Each combined-generator MPO builder is selected by the coupling graph of
# the Hamiltonian, so ring and star topologies exercise code paths that the
# chain and general Hamiltonians above never reach.

H2_ham_ring = np.zeros([nsite, nsite])
for _i in range(nsite - 1):
    H2_ham_ring[_i, _i + 1] = 40
    H2_ham_ring[_i + 1, _i] = 40
# The bond that closes the ring carries a different amplitude from the chain
# bonds. The ring builder opens that bond at site 0, relays it the length of
# the chain and closes it at the last site against H[0, n-1] and H[n-1, 0];
# giving it the neighbour amplitude would let a builder that picked up the
# wrong entry, or swapped the two directions, still pass.
H2_ham_ring[0, nsite - 1] = 25
H2_ham_ring[nsite - 1, 0] = 25

# Hub at site 1 rather than site 0, so the builder's leaves-left and
# leaves-right branches are both used.
H2_ham_star = np.zeros([nsite, nsite])
for _i in range(nsite):
    if _i != 1:
        H2_ham_star[1, _i] = 30
        H2_ham_star[_i, 1] = 30


def _run_hops_topology(H2_ham_topology, method=None, bond_dim_max=40,
                       mps_epsilon=1e-13, flag_mpo_optimize=True):
    """Runs vector HOPS (method=None) or tensor HOPS for a topology.

    The MPS budget is tighter than elsewhere in this module because the ring
    closes an extra bond and so entangles more than the chain: at
    BOND_DIM_MAX=20 / MPS_EPSILON=1e-10 the ring trajectory sits ~6e-8 from
    vector HOPS through MPS truncation alone, for the general MPO path as
    well as for the combined one. These settings put the wavefunction error
    below the level being tested so the comparison probes the MPO.
    """
    sys_param_topology = dict(sys_param)
    sys_param_topology['HAMILTONIAN'] = np.array(
        H2_ham_topology, dtype=np.complex128,
    )
    if method is None:
        traj_vector = HOPS(
            sys_param_topology,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
        traj_vector.initialize(psi_0)
        traj_vector.propagate(t_max, t_step)
        return np.array(traj_vector.storage['psi_traj'])
    traj_tensor = HopsTensorTrajectory(
        system_param=sys_param_topology,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param={
            'MPS_EPSILON': mps_epsilon, 'METHOD': method,
            'BOND_DIM_MAX': bond_dim_max,
            'FLAG_MPO_OPTIMIZE': flag_mpo_optimize,
        },
    )
    traj_tensor.initialize(psi_0)
    traj_tensor.propagate(t_max, t_step)
    return np.array(traj_tensor.storage['psi_traj'])


# ------------------------------------------------------------
# TEST: Ring and star statenumber MPOs match vector HOPS
# ------------------------------------------------------------
@pytest.mark.level(2)
@pytest.mark.parametrize('topology', ['ring', 'star'])
@pytest.mark.parametrize('flag_mpo_optimize', [True, False])
def test_statenumber_topology_matches_vector_hops(topology, flag_mpo_optimize):
    # This case tests that ring and star Hamiltonians reproduce vector HOPS
    # both through their combined-generator MPOs (FLAG_MPO_OPTIMIZE True) and
    # through the general hierarchy-plus-Hamiltonian path (False). Both are
    # long-ranged Hamiltonians that the nearest-neighbor MPO cannot
    # represent, and the combined builders reach them at a bond dimension
    # fixed by the topology rather than by the interaction range.
    H2_ham_topology = {
        'ring': H2_ham_ring, 'star': H2_ham_star,
    }[topology]
    psi_vector = _run_hops_topology(H2_ham_topology)
    psi_tensor = _run_hops_topology(
        H2_ham_topology, method='number',
        flag_mpo_optimize=flag_mpo_optimize,
    )

    assert len(psi_vector) == len(psi_tensor), (
        'Trajectory lengths differ'
    )
    max_err = max(
        np.linalg.norm(psi_tensor[i] - psi_vector[i])
        for i in range(len(psi_vector))
    )
    # Largest error observed across the two topologies is 3.8e-12 with the
    # optimization on and 2.2e-12 with it off.
    assert max_err < 1e-11, (
        f'Statenumber {topology} tensor HOPS diverged from vector HOPS: '
        f'max wf error = {max_err:.2e}'
    )
