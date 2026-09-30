import numpy as np
import pytest
import scipy as sp
from scipy import sparse
from types import SimpleNamespace

from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_tensor_trajectory import HopsTensorTrajectory
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.nondyadic_spectroscopy import (
    _apply_op_and_track_norm,
    _build_operators_abs,
    _build_operators_fluor,
    _spectroscopy_key,
    SpectroscopyDispatch,
    calc_absorption_response,
    calc_fluorescence_response,
)

__title__ = 'test_nondyadic_spectroscopy'
__author__ = 'A. Hartzell'
__maintainer__ = 'A. Hartzell'


def _make_traj(n_site, seed=0, eom='NONLINEAR'):
    """
    Builds an uninitialized HopsTrajectory with the given number of
    chromophore sites (total states = n_site + 1).
    """
    n_state = n_site + 1

    H2_sys = np.zeros((n_state, n_state), dtype=np.complex128)
    if n_site == 1:
        H2_sys[1, 1] = 100.0
    else:
        H2_exc = np.diag([100.0, 0.0][:n_site]) + np.diag(
            [-50.0] * (n_site - 1), k=1
        ) + np.diag([-50.0] * (n_site - 1), k=-1)
        H2_sys[1:, 1:] = H2_exc

    list_lop = []
    for i in range(n_site):
        lop = np.zeros((n_state, n_state), dtype=np.float64)
        lop[i + 1, i + 1] = 1.0
        list_lop.append(lop)

    gw_sysbath = [[10.0, 10.0]] * n_site

    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': gw_sysbath,
        'L_HIER': list_lop,
        'L_NOISE1': list_lop,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_sysbath,
    }
    dict_noise_param = {
        'SEED': seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 200.0,
        'TAU': 1.0,
    }
    dict_hier_param = {'MAXHIER': 2}
    dict_eom_param = {'EQUATION_OF_MOTION': eom}

    return HOPS(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param=dict_hier_param,
        eom_param=dict_eom_param,
    )


def _make_dyadic_params(n_site, t_max, t_step, max_hier, seed):
    """
    Builds shared physical parameters for dyadic vs nondyadic comparisons.
    Returns the Hamiltonian, dipoles, field, bath modes, l-operators, and
    the chromophore/convergence/noise dicts needed by both sides.
    """
    from mesohops.trajectory.dyadic_spectra import (
        prepare_chromophore_input_dict,
        prepare_convergence_parameter_dict,
    )
    from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp

    n_state = n_site + 1

    H2_sys = np.zeros((n_state, n_state), dtype=np.complex128)
    H2_exc = np.diag([100.0, 0.0][:n_site]) + np.diag(
        [-50.0] * (n_site - 1), k=1
    ) + np.diag([-50.0] * (n_site - 1), k=-1)
    H2_sys[1:, 1:] = H2_exc

    list_transition_dipoles = np.array([[0.6, 0.0, 0.3], [0.0, 0.5, 0.4]])
    E_1 = np.array([0.0, 0.0, 1.0])

    list_modes = bcf_convert_dl_to_exp(50.0, 50.0, 300.0)
    list_lop = [
        sparse.coo_matrix(([1], ([i + 1], [i + 1])), shape=(n_state, n_state))
        for i in range(n_site)
    ]

    chromophore_dict = prepare_chromophore_input_dict(
        list_transition_dipoles, H2_sys, {'list_lop': list_lop, 'list_modes': list_modes}
    )
    convergence_dict = prepare_convergence_parameter_dict(
        t_step=t_step, max_hier=max_hier
    )

    t_total = 1000.0 + t_max
    dict_noise_param = {
        'SEED': seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': t_total,
        'TAU': 0.5,
    }

    return (H2_sys, list_transition_dipoles, E_1, n_state, n_site,
            chromophore_dict, convergence_dict, dict_noise_param)


# ============================================================
# TEST SUITE: _check_eom()
# ============================================================

# ------------------------------------------------------------
# TEST: rejects NORMALIZED NONLINEAR EOM
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_eom_rejects_normalized_nonlinear_abs():
    # This case tests that absorption raises for NORMALIZED NONLINEAR
    traj = _make_traj(2, eom='NORMALIZED NONLINEAR')
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(UnsupportedRequest, match='NORMALIZED NONLINEAR'):
        calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)


@pytest.mark.level(1)
def test_check_eom_rejects_normalized_nonlinear_fluor():
    # This case tests that fluorescence raises for NORMALIZED NONLINEAR
    traj = _make_traj(2, eom='NORMALIZED NONLINEAR')
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    with pytest.raises(UnsupportedRequest, match='NORMALIZED NONLINEAR'):
        calc_fluorescence_response(traj, list_transition_dipoles, E_1, E_sig, 4.0, 6.0, 2.0)


# ------------------------------------------------------------
# TEST: rejects initialized trajectory
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_eom_rejects_initialized_traj_abs():
    # This case tests that absorption raises if trajectory is already initialized
    traj = _make_traj(2)
    n_state = 3
    P1_psi_0 = np.zeros(n_state, dtype=np.complex128)
    P1_psi_0[0] = 1.0
    traj.initialize(P1_psi_0)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='initialized trajectory'):
        calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)


@pytest.mark.level(1)
def test_check_eom_rejects_initialized_traj_fluor():
    # This case tests that fluorescence raises if trajectory is already initialized
    traj = _make_traj(2)
    n_state = 3
    P1_psi_0 = np.zeros(n_state, dtype=np.complex128)
    P1_psi_0[0] = 1.0
    traj.initialize(P1_psi_0)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='initialized trajectory'):
        calc_fluorescence_response(traj, list_transition_dipoles, E_1, E_sig, 4.0, 6.0, 2.0)


# ============================================================
# TEST SUITE: _build_operators_abs()
# ============================================================

# ------------------------------------------------------------
# TEST: raise operator preserves ground state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_operators_abs_raise_keeps_ground():
    # This case tests that O2_raise has a 1 at (0, 0) to preserve |g>
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    O2_raise, _ = _build_operators_abs(list_transition_dipoles, E_1)
    O2_dense = O2_raise.toarray()
    n_total = O2_dense.shape[0]
    # Analytical: ground row preserves ground state
    assert O2_dense[0, 0] == 1.0
    # This case tests that ground row has no off-diagonal entries
    for j in range(1, n_total):
        assert O2_dense[0, j] == 0.0, f'O2_raise[0,{j}] should be 0'
    # This case tests that excited rows have entries only in column 0
    for i in range(1, n_total):
        for j in range(1, n_total):
            assert O2_dense[i, j] == 0.0, f'O2_raise[{i},{j}] should be 0'


# ------------------------------------------------------------
# TEST: raise operator has correct excited entries
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_operators_abs_raise_excited_entries():
    # This case tests that (i+1, 0) entries equal mu_i . E
    list_transition_dipoles = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    O2_raise, _ = _build_operators_abs(list_transition_dipoles, E_1)
    O2_dense = O2_raise.toarray()
    # Analytical: O2_raise[i+1, 0] = mu_i . E_1
    # list_transition_dipoles = [[3,0,0],[0,5,0]], E_1 = [1,0,0]
    # mu_0 . E_1 = 3.0, mu_1 . E_1 = 0.0
    assert O2_dense[1, 0] == pytest.approx(3.0)
    assert O2_dense[2, 0] == pytest.approx(0.0)


# ------------------------------------------------------------
# TEST: F2 response operator structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_operators_abs_F2_structure():
    # This case tests that F2 has nonzero entries only in row 0, columns 1:
    list_transition_dipoles = np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    _, F2_dense = _build_operators_abs(list_transition_dipoles, E_1)
    assert F2_dense[0, 0] == 0.0
    assert F2_dense[0, 1] == 2.0
    assert F2_dense[0, 2] == 0.0
    assert np.all(F2_dense[1:, :] == 0.0)


# ============================================================
# TEST SUITE: _build_operators_fluor()
# ============================================================

# ------------------------------------------------------------
# TEST: Fluorescence operators have correct structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_operators_fluor_structure():
    # This case tests that the raising, lowering+identity, and response
    # operators constructed for fluorescence have analytically correct entries.
    # list_transition_dipoles = [[1,0,0],[0,1,0]], E_1 = [1,0,0], E_sig = [0,1,0]
    # mu_0.E_1 = 1.0, mu_1.E_1 = 0.0
    # mu_0.E_sig = 0.0, mu_1.E_sig = 1.0
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([0.0, 1.0, 0.0])
    O2_raise, O2_lower_ident, F2 = _build_operators_fluor(list_transition_dipoles, E_1, E_sig)
    O2_raise_dense = O2_raise.toarray()
    # Analytical: O2_raise[i+1, 0] = mu_i . E_1
    assert O2_raise_dense[1, 0] == pytest.approx(1.0)  # mu_0 . E_1 = 1.0
    assert O2_raise_dense[2, 0] == pytest.approx(0.0)  # mu_1 . E_1 = 0.0
    # Analytical: O2_lower_ident has |g><e_i| weighted by mu_i . E_sig
    O2_lower_dense = O2_lower_ident.toarray()
    assert O2_lower_dense[0, 1] == pytest.approx(0.0)  # mu_0 . E_sig = 0.0
    assert O2_lower_dense[0, 2] == pytest.approx(1.0)  # mu_1 . E_sig = 1.0
    # Excited diagonal identity
    assert O2_lower_dense[1, 1] == pytest.approx(1.0)
    assert O2_lower_dense[2, 2] == pytest.approx(1.0)
    # Analytical: F2[0, i+1] = mu_i . E_sig
    assert F2[0, 1] == pytest.approx(0.0)
    assert F2[0, 2] == pytest.approx(1.0)
    # Verify all other elements are zero
    assert O2_raise_dense[0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(O2_raise_dense[1:, 1:], 0.0, atol=1e-15)
    np.testing.assert_allclose(O2_lower_dense[1:, 0], 0.0, atol=1e-15)
    assert O2_lower_dense[0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(F2[1:, :], 0.0, atol=1e-15)
    assert F2[0, 0] == pytest.approx(0.0)


# ------------------------------------------------------------
# TEST: Non-orthogonal dipoles produce correct operator elements
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_operators_abs_nonorthogonal_dipoles():
    # This case tests that non-axis-aligned dipole moments produce the
    # correct operator entries, catching bugs that only appear with
    # non-trivial dot products (not exactly 0 or 1).
    list_transition_dipoles = np.array([[0.6, 0.0, 0.8], [0.3, 0.4, 0.0]])
    E_1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    O2_raise, F2 = _build_operators_abs(list_transition_dipoles, E_1)
    O2_dense = O2_raise.toarray()
    # Analytical: mu_0 . E_1 = (0.6 + 0.0) / sqrt(2) = 0.6/sqrt(2)
    #             mu_1 . E_1 = (0.3 + 0.4) / sqrt(2) = 0.7/sqrt(2)
    mu0_dot_E = (0.6 + 0.0) / np.sqrt(2)
    mu1_dot_E = (0.3 + 0.4) / np.sqrt(2)
    np.testing.assert_allclose(O2_dense[1, 0], mu0_dot_E, atol=1e-14)
    np.testing.assert_allclose(O2_dense[2, 0], mu1_dot_E, atol=1e-14)
    np.testing.assert_allclose(O2_dense[0, 0], 1.0, atol=1e-14)  # ground identity
    np.testing.assert_allclose(F2[0, 1], mu0_dot_E, atol=1e-14)
    np.testing.assert_allclose(F2[0, 2], mu1_dot_E, atol=1e-14)


# ============================================================
# TEST SUITE: _apply_op_and_track_norm()
# ============================================================

# ------------------------------------------------------------
# TEST: Identity operator gives norm ratio of 1.0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_op_and_track_norm_identity():
    # This case tests that applying the identity operator preserves norm,
    # so the returned coefficient must be exactly 1.0.
    n_site = 2
    n_state = n_site + 1
    traj = _make_traj(n_site, seed=0)
    P1_psi_0 = np.zeros(n_state, dtype=np.complex128)
    P1_psi_0[0] = 1.0
    traj.initialize(P1_psi_0)
    I2 = sp.sparse.eye(n_state, format='coo')
    norm_ratio = _apply_op_and_track_norm(
        traj,
        lambda: traj._operator(I2.toarray()),
        SpectroscopyDispatch('vector', None, 'embedded', 'NL'),
    )
    # Analytical: identity preserves norm, so ratio = 1.0
    np.testing.assert_allclose(norm_ratio, 1.0, atol=1e-10)


# ------------------------------------------------------------
# TEST: Nontrivial operator gives the expected norm ratio
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_op_and_track_norm_operator_ratio():
    # This case checks that the helper measures the actual pre/post
    # norm change, not just the identity case. The operator maps
    # [1, 1, 0] -> [3, 1, 0], so the norm ratio is 10 / 2 = 5.
    n_site = 2
    n_state = n_site + 1
    traj = _make_traj(n_site, seed=0)
    P1_psi_0 = np.zeros(n_state, dtype=np.complex128)
    P1_psi_0[:] = np.array([1.0, 1.0, 0.0], dtype=np.complex128)
    traj.initialize(P1_psi_0)
    op = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.complex128)
    norm_ratio = _apply_op_and_track_norm(
        traj,
        lambda: traj._operator(op),
        SpectroscopyDispatch('vector', None, 'embedded', 'NL'),
    )
    np.testing.assert_allclose(norm_ratio, 5.0, atol=1e-10)


class _FakeVacuumTrajectory:
    def __init__(self, full_state):
        self._full_state = np.array(full_state, dtype=np.complex128)
        self.basis = SimpleNamespace(
            eom=SimpleNamespace(param={'EQUATION_OF_MOTION': 'NONLINEAR'})
        )
        self.wavefunction = self

    @property
    def psi(self):
        return self._full_state[1:]

    @property
    def manifold_norm_sq(self):
        return np.sum(np.conj(self._full_state) * self._full_state).real

    def _operator(self, op):
        self._full_state = op @ self._full_state


# ------------------------------------------------------------
# TEST: Vacuum branch uses manifold norm rather than psi norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_op_and_track_norm_vacuum_branch():
    # This case ensures the vacuum-convention branch uses the manifold
    # norm helper before and after the operator application.
    traj = _FakeVacuumTrajectory([1.0, 1.0, 0.0])
    op = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.complex128)
    norm_ratio = _apply_op_and_track_norm(
        traj,
        lambda: traj._operator(op),
        SpectroscopyDispatch('tensor', 'number', 'vacuum', 'NL'),
    )
    # Full-state norm changes from ||[1,1,0]||^2 = 2 to ||[2,1,0]||^2 = 5.
    np.testing.assert_allclose(norm_ratio, 2.5, atol=1e-10)


# ============================================================
# TEST SUITE: calc_absorption_response()
# ============================================================

# ------------------------------------------------------------
# TEST: correct output shape
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_shape():
    # This case tests output length matches number of propagation time steps
    traj = _make_traj(2, seed=42)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_absorption_response(traj, list_transition_dipoles, E_1, 10.0, 2.0)
    assert len(C1_corr_t) == 5


# ------------------------------------------------------------
# TEST: deterministic with same seed
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_deterministic():
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])

    traj1 = _make_traj(2, seed=7)
    C1_a = calc_absorption_response(traj1, list_transition_dipoles, E_1, 10.0, 2.0)

    traj2 = _make_traj(2, seed=7)
    C1_b = calc_absorption_response(traj2, list_transition_dipoles, E_1, 10.0, 2.0)

    np.testing.assert_allclose(C1_a, C1_b)


# ------------------------------------------------------------
# TEST: C(0) is nonzero
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_c0_value():
    traj = _make_traj(2, seed=0)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)
    # Analytical: C(t_step) ≈ 2 * sum_i |mu_i . E_1|^2 for short times.
    # Factor of 2 accounts for the complex conjugate pathway.
    # list_transition_dipoles = [[1,0,0],[0,1,0]], E_1 = [1,0,0]
    # mu_0 . E_1 = 1.0, mu_1 . E_1 = 0.0
    # C(t_step) ≈ 2 * (|1|^2 + |0|^2) = 2.0 within stochastic tolerance
    np.testing.assert_allclose(
        abs(C1_corr_t[0]), 2.0, atol=1e-2,
        err_msg='C(t_step) should be close to 2 * sum(|mu_i . E|^2) = 2.0',
    )


# ------------------------------------------------------------
# TEST: single-site system (n_site=1)
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_single_site():
    traj = _make_traj(1, seed=0)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)
    assert len(C1_corr_t) == 2
    assert abs(C1_corr_t[0]) > 0.0


# ------------------------------------------------------------
# TEST: LINEAR EOM short-time amplitude
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_linear_c0_value():
    # Locks in the LINEAR readout convention (see _readout_prefactor).
    # Under LINEAR psi[0]=1 exactly, so <psi|F|psi>(t_step) ≈ |mu|^2
    # and the function should return ≈ 2*|mu|^2 = 2.0 for this dipole
    # choice. A regression that re-applied the dyadic
    # dyadic normalization prefactor under LINEAR would return ≈ 4.0
    # (extra (1+|mu|^2) factor at t≈0).
    traj = _make_traj(2, seed=0, eom='LINEAR')
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)
    np.testing.assert_allclose(
        abs(C1_corr_t[0]), 2.0, atol=1e-2,
        err_msg='LINEAR C(t_step) should match NONLINEAR analytical limit '
                '2 * |mu|^2 = 2.0; a (1+|mu|^2) prefactor leak would give 4.0',
    )


# ------------------------------------------------------------
# TEST: LINEAR readout matches the analytical correlator form
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_linear_correlator_form():
    # The (N+1)-embedded operators decouple |g> from the bath in H
    # and in every L (operator-construction property, true under all
    # EOMs). Under LINEAR — raw noise, no rescaling term — psi(t)[0]
    # = 1 holds deterministically, so <psi|F|psi> reduces to
    # sum_i (mu_i.E) * psi(t)[i+1], the absorption correlator
    # directly. The function should return 2x this with no extra
    # norm machinery. Catches any regression that re-applies the
    # dyadic normalization prefactor (or any extra scalar factor) under
    # LINEAR, at machine precision with no ensemble floor.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    traj = _make_traj(2, seed=7, eom='LINEAR')
    C1_func = calc_absorption_response(traj, list_transition_dipoles, E_1, 20.0, 2.0)

    # psi[0] = 1 invariant under LINEAR — structural property of the setup
    psi_traj = np.asarray(traj.storage['psi_traj'])
    np.testing.assert_allclose(
        psi_traj[:, 0], 1.0, atol=1e-10,
        err_msg='LINEAR should hold psi[0]=1 (embedded |g> decoupling + '
                'no rescaling term)',
    )

    # Manual correlator: 2 * sum_i (mu_i.E) * psi(t)[i+1]
    list_mu_dot_E = list_transition_dipoles @ E_1
    C1_manual = 2 * (psi_traj[1:, 1:] @ list_mu_dot_E)
    np.testing.assert_allclose(C1_func, C1_manual, atol=1e-10)


# ------------------------------------------------------------
# TEST: psi[0]=1 invariant under LINEAR and plain NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_psi_g_invariant_under_linear_and_nonlinear():
    # Structural property of the (N+1)-embedded picture: |g> is
    # decoupled from the bath in H and in every L. Combined with
    # the absence of any global-rescaling term (which only
    # NORMALIZED NONLINEAR has), this fixes psi[0]=1 deterministically
    # under BOTH LINEAR and plain NONLINEAR. The two EOMs differ in
    # the noise measure (raw vs Girsanov-shifted), not in psi[0]
    # dynamics — pinning this for both guards against any future
    # change to H, the L construction, or the raise/lower operators
    # that would silently couple |g> to the bath and break the
    # readout invariants both branches of _readout_prefactor depend
    # on.
    list_transition_dipoles = np.array([[0.6, 0.0, 0.3], [0.0, 0.5, 0.4]])
    E_1 = np.array([0.0, 0.0, 1.0])
    for eom in ('LINEAR', 'NONLINEAR'):
        traj = _make_traj(2, seed=7, eom=eom)
        calc_absorption_response(traj, list_transition_dipoles, E_1, 20.0, 2.0)
        psi_traj = np.asarray(traj.storage['psi_traj'])
        np.testing.assert_allclose(
            psi_traj[:, 0], 1.0, atol=1e-10,
            err_msg=f'{eom}: psi[0] should stay at 1 in the embedded picture',
        )


# ------------------------------------------------------------
# TEST: NORMALIZED NONLINEAR is a uniform rescaling of plain NONLINEAR
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_normalized_nonlinear_rescales_plain_nonlinear():
    # Central claim of the corrected readout rationale: under the
    # same noise seed, ψ_NORM(t) = c(t) · ψ_NL(t) where c(t) is a
    # complex scalar with |c(t)|^2 equal to the post-raise norm ratio
    # divided by ||ψ_NL(t)||^2. This
    # holds because operator_expectation in the EOM is normalized
    # (divides by <ψ|ψ>), so <L> is rescale-invariant and the
    # NL/NORM-NL EOMs differ only by a global scale factor c(t)
    # satisfying dc/dt = -norm_corr · c. The rescaling collapses the
    # dyadic readout normalization prefactor times <ψ|F|ψ>/||ψ||^2 to the same per-realization
    # value for both EOMs, which is what makes the existing dyadic-
    # vs-nondyadic comparison pass at rtol=5e-3 and what justifies
    # using the dyadic prefactor for plain NL inside _readout_prefactor.
    # Pinning this identity guards against any change to
    # operator_expectation or the EOM that would silently break it.
    n_site = 2
    n_state = n_site + 1
    seed = 7
    t_max, t_step = 10.0, 2.0

    list_transition_dipoles = np.array([[0.6, 0.0, 0.3], [0.0, 0.5, 0.4]])
    E_1 = np.array([0.0, 0.0, 1.0])
    O2_raise, _ = _build_operators_abs(list_transition_dipoles, E_1)

    P1_psi_0 = np.zeros(n_state, dtype=np.complex128)
    P1_psi_0[0] = 1.0

    traj_nl = _make_traj(n_site, seed=seed, eom='NONLINEAR')
    traj_nl.initialize(P1_psi_0)
    traj_nl._operator(O2_raise.toarray())
    traj_nl.propagate(t_max, t_step)
    psi_nl = np.asarray(traj_nl.storage['psi_traj'])

    traj_norm = _make_traj(n_site, seed=seed, eom='NORMALIZED NONLINEAR')
    traj_norm.initialize(P1_psi_0)
    traj_norm._operator(O2_raise.toarray())
    traj_norm.propagate(t_max, t_step)
    psi_norm = np.asarray(traj_norm.storage['psi_traj'])

    # ψ_NL[t, 0] = 1 (|g> decoupled, no rescaling under plain NL), so
    # c(t) = ψ_NORM[t, 0] / ψ_NL[t, 0] = ψ_NORM[t, 0]. Verify all
    # other components rescale by the same scalar.
    coeff_traj = psi_norm[:, 0]
    psi_predicted = coeff_traj[:, None] * psi_nl
    np.testing.assert_allclose(
        psi_norm, psi_predicted, atol=1e-6,
        err_msg='NORMALIZED NL should be a uniform (scalar) rescaling '
                'of plain NL — non-uniformity means <L> is no longer '
                'rescale-invariant',
    )

    # Norm-preservation under NORM-NL plus the rescaling identity
    # forces |c(t)|^2 = post-raise norm ratio / ||ψ_NL(t)||^2.
    coeff_post_op = np.dot(np.conj(psi_nl[0]), psi_nl[0]).real
    list_norm_sq_nl = np.sum(np.conj(psi_nl) * psi_nl, axis=1).real
    np.testing.assert_allclose(
        np.abs(coeff_traj) ** 2, coeff_post_op / list_norm_sq_nl, atol=1e-6,
        err_msg='|c(t)|^2 should equal the post-raise norm ratio / ||ψ_NL(t)||^2',
    )


# ============================================================
# TEST SUITE: calc_fluorescence_response()
# ============================================================

# ------------------------------------------------------------
# TEST: correct output shape
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_fluorescence_response_shape():
    traj = _make_traj(2, seed=42)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_fluorescence_response(
        traj, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0
    )
    assert len(C1_corr_t) == 5


# ------------------------------------------------------------
# TEST: deterministic with same seed
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_fluorescence_response_deterministic():
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])

    traj1 = _make_traj(2, seed=7)
    C1_a = calc_fluorescence_response(traj1, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0)

    traj2 = _make_traj(2, seed=7)
    C1_b = calc_fluorescence_response(traj2, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0)

    np.testing.assert_allclose(C1_a, C1_b)


# ------------------------------------------------------------
# TEST: non-trivial dipole weighting produces nonzero result
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_fluorescence_response_two_coefficients():
    traj = _make_traj(2, seed=0)
    list_transition_dipoles = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([0.0, 1.0, 0.0])
    C1_corr_t = calc_fluorescence_response(
        traj, list_transition_dipoles, E_1, E_sig, 4.0, 6.0, 2.0
    )
    # Invariant: fluorescence response should be complex and non-trivial
    assert np.iscomplexobj(C1_corr_t), 'Response should be complex'
    assert np.any(np.abs(C1_corr_t) > 1e-12), (
        'Fluorescence response should be nonzero for nonzero dipoles'
    )


# ------------------------------------------------------------
# TEST: LINEAR fluorescence readout matches analytical correlator
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_fluorescence_response_linear_correlator_form():
    # The (N+1)-embedded operators decouple |g> from the bath under
    # all EOMs, and LINEAR has no global-rescaling term, so after
    # the lowering+identity operator at the end of t2 lifts psi[0]
    # to c_g, psi[0] stays at c_g exactly through the t3 phase. The
    # detection readout therefore reduces to
    # 4 * conj(c_g) * sum_i (mu_i.E_sig) * psi(t3)[i+1]. Catches
    # any regression that re-applies the dyadic normalization prefactor
    # (or any extra scalar factor) under LINEAR.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    t2 = 4.0
    t3_max = 10.0
    t_step = 2.0
    traj = _make_traj(2, seed=7, eom='LINEAR')
    C1_func = calc_fluorescence_response(
        traj, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step
    )

    # Pull the stored wavefunction trajectory; the detection-phase
    # window starts one step after de-excitation (t = t2 + t_step).
    psi_traj = np.asarray(traj.storage['psi_traj'])
    idx_t2 = round(t2 / t_step)
    psi_t3 = psi_traj[idx_t2 + 1:]

    # |g> decoupled under LINEAR -> psi[0] constant during t3
    np.testing.assert_allclose(
        psi_t3[:, 0], psi_t3[0, 0], atol=1e-10,
        err_msg='psi[0] should be constant during t3 under LINEAR',
    )

    # Manual correlator: 4 * conj(psi[0]) * sum_i (mu_i.E_sig) * psi[i+1]
    list_mu_dot_Esig = list_transition_dipoles @ E_sig
    C1_manual = 4 * np.conj(psi_t3[:, 0]) * (psi_t3[:, 1:] @ list_mu_dot_Esig)
    np.testing.assert_allclose(C1_func, C1_manual, atol=1e-10)


# ============================================================
# TEST SUITE: dyadic vs nondyadic comparison
# ============================================================

# ------------------------------------------------------------
# TEST: absorption C(t) matches DyadicSpectra
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_absorption_matches_dyadic_spectra():
    """
    Compares non-dyadic absorption against DyadicSpectra with identical
    physical parameters and the same noise seed. Dipoles are scaled so
    S = sum((mu_i.E)^2) = 1. The dyadic formulation uses NORMALIZED
    NONLINEAR while the nondyadic uses NONLINEAR, so a small tolerance
    accounts for normalization drift.
    """
    from mesohops.trajectory.dyadic_spectra import (
        DyadicSpectra as DHOPS,
    )
    from mesohops.trajectory.dyadic_spectra import (
        prepare_spectroscopy_input_dict,
    )

    n_site = 2
    seed = 42
    t_max = 50.0
    t_step = 1.0
    max_hier = 4

    (H2_sys, list_transition_dipoles, E_1, n_state, n_site,
     chromophore_dict, convergence_dict, dict_noise_param) = _make_dyadic_params(
        n_site, t_max, t_step, max_hier, seed
    )

    # Dyadic calculation
    spec_dict = prepare_spectroscopy_input_dict(
        'ABSORPTION',
        {'t_1': t_max},
        {'E_1': E_1},
        {'list_ket_sites': np.arange(1, n_site + 1)},
    )
    dhops = DHOPS(spec_dict, chromophore_dict, convergence_dict, seed)
    C1_dyadic = dhops.calculate_spectrum()

    # Non-dyadic calculation with matching parameters
    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': chromophore_dict['gw_sysbath_hier'],
        'L_HIER': chromophore_dict['lop_list_hier'],
        'L_NOISE1': chromophore_dict['lop_list_noise'],
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': chromophore_dict['gw_sysbath_noise'],
    }
    traj = HOPS(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param={'MAXHIER': max_hier},
        eom_param={'EQUATION_OF_MOTION': 'NONLINEAR'},
    )
    C1_nondyadic = calc_absorption_response(traj, list_transition_dipoles, E_1, t_max, t_step)

    np.testing.assert_allclose(C1_nondyadic, C1_dyadic, rtol=5e-3)


# ------------------------------------------------------------
# TEST: fluorescence C(t) matches DyadicSpectra
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fluorescence_matches_dyadic_spectra():
    """
    Compares non-dyadic fluorescence against DyadicSpectra with identical
    physical parameters and the same noise seed.
    """
    from mesohops.trajectory.dyadic_spectra import (
        DyadicSpectra as DHOPS,
    )
    from mesohops.trajectory.dyadic_spectra import (
        prepare_spectroscopy_input_dict,
    )

    n_site = 2
    seed = 42
    t2 = 50.0
    t3_max = 50.0
    t_step = 1.0
    max_hier = 4

    (H2_sys, list_transition_dipoles, E_1, n_state, n_site,
     chromophore_dict, convergence_dict, dict_noise_param) = _make_dyadic_params(
        n_site, t2 + t3_max, t_step, max_hier, seed
    )
    E_sig = E_1

    # Dyadic calculation
    spec_dict = prepare_spectroscopy_input_dict(
        'FLUORESCENCE',
        {'t_2': t2, 't_3': t3_max},
        {'E_1': E_1, 'E_sig': E_sig},
        {
            'list_ket_sites': np.arange(1, n_site + 1),
            'list_bra_sites': np.arange(1, n_site + 1),
        },
    )
    dhops = DHOPS(spec_dict, chromophore_dict, convergence_dict, seed)
    C1_dyadic = dhops.calculate_spectrum()

    # Non-dyadic calculation with matching parameters
    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': chromophore_dict['gw_sysbath_hier'],
        'L_HIER': chromophore_dict['lop_list_hier'],
        'L_NOISE1': chromophore_dict['lop_list_noise'],
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': chromophore_dict['gw_sysbath_noise'],
    }
    traj = HOPS(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param={'MAXHIER': max_hier},
        eom_param={'EQUATION_OF_MOTION': 'NONLINEAR'},
    )
    C1_nondyadic = calc_fluorescence_response(
        traj, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step
    )

    np.testing.assert_allclose(C1_nondyadic, C1_dyadic, rtol=5e-3)


# ============================================================
# TEST SUITE: HopsTensorTrajectory compatibility
# ============================================================

def _make_tensor_traj(n_site, seed=0, eom='NONLINEAR'):
    """
    Builds an uninitialized HopsTensorTrajectory with the given number
    of chromophore sites (total states = n_site + 1).
    """
    n_state = n_site + 1

    H2_sys = np.zeros((n_state, n_state), dtype=np.complex128)
    H2_exc = np.diag([100.0, 0.0][:n_site]) + np.diag(
        [-50.0] * (n_site - 1), k=1
    ) + np.diag([-50.0] * (n_site - 1), k=-1)
    H2_sys[1:, 1:] = H2_exc

    list_lop = []
    for i in range(n_site):
        lop = np.zeros((n_state, n_state), dtype=np.float64)
        lop[i + 1, i + 1] = 1.0
        list_lop.append(lop)

    gw_sysbath = [[10.0, 10.0]] * n_site

    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': gw_sysbath,
        'L_HIER': list_lop,
        'L_NOISE1': list_lop,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_sysbath,
    }
    dict_noise_param = {
        'SEED': seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 200.0,
        'TAU': 1.0,
    }
    dict_hier_param = {'MAXHIER': 2}
    dict_eom_param = {'EQUATION_OF_MOTION': eom}
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }

    return HopsTensorTrajectory(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param=dict_hier_param,
        eom_param=dict_eom_param,
        tensor_param=tensor_param,
    )


# ------------------------------------------------------------
# TEST: tensor absorption produces correct shape
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_absorption_shape():
    # This case tests that calc_absorption_response works with
    # HopsTensorTrajectory and returns the correct output length.
    traj = _make_tensor_traj(2, seed=42)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_absorption_response(traj, list_transition_dipoles, E_1, 10.0, 2.0)
    assert len(C1_corr_t) == 5


# ------------------------------------------------------------
# TEST: tensor and vector absorption agree
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_absorption_matches_vector():
    # This case tests that tensor and vector trajectories produce
    # the same absorption C(t) for the same seed and parameters.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])

    traj_vec = _make_traj(2, seed=7)
    C1_vec = calc_absorption_response(traj_vec, list_transition_dipoles, E_1, 10.0, 2.0)

    traj_tensor = _make_tensor_traj(2, seed=7)
    C1_tensor = calc_absorption_response(traj_tensor, list_transition_dipoles, E_1, 10.0, 2.0)

    np.testing.assert_allclose(C1_tensor, C1_vec, atol=1e-6)


# ------------------------------------------------------------
# TEST: tensor fluorescence produces correct shape
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_fluorescence_shape():
    # This case tests that calc_fluorescence_response works with
    # HopsTensorTrajectory and returns the correct output length.
    traj = _make_tensor_traj(2, seed=42)
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    C1_corr_t = calc_fluorescence_response(
        traj, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0
    )
    assert len(C1_corr_t) == 5


# ------------------------------------------------------------
# TEST: tensor and vector fluorescence agree
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_fluorescence_matches_vector():
    # This case tests that tensor and vector trajectories produce
    # the same fluorescence C(t) for the same seed and parameters.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])

    traj_vec = _make_traj(2, seed=7)
    C1_vec = calc_fluorescence_response(traj_vec, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0)

    traj_tensor = _make_tensor_traj(2, seed=7)
    C1_tensor = calc_fluorescence_response(
        traj_tensor, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0
    )

    np.testing.assert_allclose(C1_tensor, C1_vec, atol=1e-6)


# ------------------------------------------------------------
# TEST: tensor LINEAR absorption per-trajectory identity
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_absorption_response_linear_correlator_form():
    # _readout_prefactor branches on the EOM string, which is
    # populated identically on HopsTrajectory and HopsTensorTrajectory.
    # Asserts the same per-realization identity as the vector LINEAR
    # test (test_calc_absorption_response_linear_correlator_form):
    # under the (N+1)-embedded picture |g> is decoupled in H and L,
    # LINEAR has no rescaling term, so psi(t)[0]=1 deterministically
    # and 2*<psi|F|psi> = 2 * sum_i (mu_i.E) * psi(t)[i+1].
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    traj = _make_tensor_traj(2, seed=7, eom='LINEAR')
    C1_func = calc_absorption_response(traj, list_transition_dipoles, E_1, 10.0, 2.0)

    psi_traj = np.asarray(traj.storage['psi_traj'])
    np.testing.assert_allclose(
        psi_traj[:, 0], 1.0, atol=1e-10,
        err_msg='LINEAR (tensor) should hold psi[0]=1',
    )

    list_mu_dot_E = list_transition_dipoles @ E_1
    C1_manual = 2 * (psi_traj[1:, 1:] @ list_mu_dot_E)
    np.testing.assert_allclose(C1_func, C1_manual, atol=1e-10)


# ------------------------------------------------------------
# TEST: tensor LINEAR fluorescence per-trajectory identity
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_fluorescence_response_linear_correlator_form():
    # Tensor analog of test_calc_fluorescence_response_linear_correlator_form.
    # After the lowering+identity operator at the end of t2, psi[0]
    # picks up a nonzero amplitude c_g and stays there during t3
    # under LINEAR (|g> decoupled in the embedded picture, no
    # rescaling term). Detection readout reduces to
    # 4 * conj(c_g) * sum_i (mu_i.E_sig) * psi(t3)[i+1].
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    t2 = 4.0
    t3_max = 10.0
    t_step = 2.0
    traj = _make_tensor_traj(2, seed=7, eom='LINEAR')
    C1_func = calc_fluorescence_response(
        traj, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step
    )

    psi_traj = np.asarray(traj.storage['psi_traj'])
    idx_t2 = round(t2 / t_step)
    psi_t3 = psi_traj[idx_t2 + 1:]

    np.testing.assert_allclose(
        psi_t3[:, 0], psi_t3[0, 0], atol=1e-10,
        err_msg='psi[0] should be constant during t3 under LINEAR (tensor)',
    )

    list_mu_dot_Esig = list_transition_dipoles @ E_sig
    C1_manual = 4 * np.conj(psi_t3[:, 0]) * (
        psi_t3[:, 1:] @ list_mu_dot_Esig
    )
    np.testing.assert_allclose(C1_func, C1_manual, atol=1e-10)


def _make_tensor_traj_sn(n_site, seed=0):
    """
    Builds an uninitialized number-representation HopsTensorTrajectory in the
    vacuum convention: NSTATES = n_site, the ground state is the all-zeros
    MPS configuration.
    """
    n_state = n_site

    H2_sys = (
        np.diag([100.0, 0.0][:n_site])
        + np.diag([-50.0] * (n_site - 1), k=1)
        + np.diag([-50.0] * (n_site - 1), k=-1)
    ).astype(np.complex128)

    list_lop = []
    for i in range(n_site):
        lop = np.zeros((n_state, n_state), dtype=np.float64)
        lop[i, i] = 1.0
        list_lop.append(lop)

    gw_sysbath = [[10.0, 10.0]] * n_site

    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': gw_sysbath,
        'L_HIER': list_lop,
        'L_NOISE1': list_lop,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_sysbath,
    }
    dict_noise_param = {
        'SEED': seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 200.0,
        'TAU': 1.0,
    }
    dict_hier_param = {'MAXHIER': 2}
    dict_eom_param = {'EQUATION_OF_MOTION': 'NONLINEAR'}
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }

    return HopsTensorTrajectory(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param=dict_hier_param,
        eom_param=dict_eom_param,
        tensor_param=tensor_param,
    )


# ------------------------------------------------------------
# TEST: statenumber absorption matches vector
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_statenumber_absorption_matches_vector():
    # CASE: Statenumber tensor trajectory should produce the same
    # absorption C(t) as vector HOPS for the same seed.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])

    traj_vec = _make_traj(2, seed=7)
    C1_vec = calc_absorption_response(traj_vec, list_transition_dipoles, E_1, 10.0, 2.0)

    traj_sn = _make_tensor_traj_sn(2, seed=7)
    C1_sn = calc_absorption_response(traj_sn, list_transition_dipoles, E_1, 10.0, 2.0)

    np.testing.assert_allclose(C1_sn, C1_vec, atol=1e-6)


# ------------------------------------------------------------
# TEST: statenumber absorption matches fullstate (tight tolerance)
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_statenumber_absorption_matches_fullstate():
    # CASE: Both representations use the same tensor EOM — differences
    # are only SVD truncation, which is negligible for this system.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])

    traj_fs = _make_tensor_traj(2, seed=7)
    C1_fs = calc_absorption_response(traj_fs, list_transition_dipoles, E_1, 10.0, 2.0)

    traj_sn = _make_tensor_traj_sn(2, seed=7)
    C1_sn = calc_absorption_response(traj_sn, list_transition_dipoles, E_1, 10.0, 2.0)

    np.testing.assert_allclose(C1_sn, C1_fs, atol=1e-10)


# ------------------------------------------------------------
# TEST: statenumber fluorescence matches vector
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_statenumber_fluorescence_matches_vector():
    # CASE: Statenumber tensor trajectory should produce the same
    # fluorescence C(t) as vector HOPS for the same seed.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])

    traj_vec = _make_traj(2, seed=7)
    C1_vec = calc_fluorescence_response(
        traj_vec, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0,
    )

    traj_sn = _make_tensor_traj_sn(2, seed=7)
    C1_sn = calc_fluorescence_response(
        traj_sn, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0,
    )

    np.testing.assert_allclose(C1_sn, C1_vec, atol=1e-6)


# ------------------------------------------------------------
# TEST: statenumber fluorescence matches fullstate (tight tolerance)
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_statenumber_fluorescence_matches_fullstate():
    # CASE: Both representations use the same tensor EOM — differences
    # are only SVD truncation, which is negligible for this system.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])

    traj_fs = _make_tensor_traj(2, seed=7)
    C1_fs = calc_fluorescence_response(
        traj_fs, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0,
    )

    traj_sn = _make_tensor_traj_sn(2, seed=7)
    C1_sn = calc_fluorescence_response(
        traj_sn, list_transition_dipoles, E_1, E_sig, 4.0, 10.0, 2.0,
    )

    np.testing.assert_allclose(C1_sn, C1_fs, atol=1e-10)

# ============================================================

def _make_traj_excited_only(n_site, seed=0, eom='LINEAR'):
    """
    Builds an uninitialized HopsTrajectory in the excited-only (N-dim)
    layout — H, L_HIER, L_NOISE1 are (n_site, n_site) with no
    embedded |g> state. Mirrors _make_traj otherwise so the per-site
    excited-block dynamics match between the two layouts under LINEAR
    with a shared seed.
    """
    n_state = n_site

    if n_site == 1:
        H2_sys = np.array([[100.0]], dtype=np.complex128)
    else:
        H2_sys = (
            np.diag([100.0, 0.0][:n_site])
            + np.diag([-50.0] * (n_site - 1), k=1)
            + np.diag([-50.0] * (n_site - 1), k=-1)
        ).astype(np.complex128)

    list_lop = []
    for i in range(n_site):
        lop = np.zeros((n_state, n_state), dtype=np.float64)
        lop[i, i] = 1.0
        list_lop.append(lop)

    gw_sysbath = [[10.0, 10.0]] * n_site

    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': gw_sysbath,
        'L_HIER': list_lop,
        'L_NOISE1': list_lop,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_sysbath,
    }
    dict_noise_param = {
        'SEED': seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 200.0,
        'TAU': 1.0,
    }
    dict_hier_param = {'MAXHIER': 2}
    dict_eom_param = {'EQUATION_OF_MOTION': eom}

    return HOPS(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param=dict_hier_param,
        eom_param=dict_eom_param,
    )


def _make_tensor_traj_excited_only(n_site, seed=0, eom='LINEAR'):
    """
    Tensor analog of _make_traj_excited_only.
    """
    n_state = n_site

    if n_site == 1:
        H2_sys = np.array([[100.0]], dtype=np.complex128)
    else:
        H2_sys = (
            np.diag([100.0, 0.0][:n_site])
            + np.diag([-50.0] * (n_site - 1), k=1)
            + np.diag([-50.0] * (n_site - 1), k=-1)
        ).astype(np.complex128)

    list_lop = []
    for i in range(n_site):
        lop = np.zeros((n_state, n_state), dtype=np.float64)
        lop[i, i] = 1.0
        list_lop.append(lop)

    gw_sysbath = [[10.0, 10.0]] * n_site

    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': gw_sysbath,
        'L_HIER': list_lop,
        'L_NOISE1': list_lop,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_sysbath,
    }
    dict_noise_param = {
        'SEED': seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 200.0,
        'TAU': 1.0,
    }
    dict_hier_param = {'MAXHIER': 2}
    dict_eom_param = {'EQUATION_OF_MOTION': eom}
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }

    return HopsTensorTrajectory(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param=dict_hier_param,
        eom_param=dict_eom_param,
        tensor_param=tensor_param,
    )


# ------------------------------------------------------------
# TEST: N-dim and (N+1)-dim LINEAR keys agree per-realization
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_calc_absorption_response_linear_keys_match():
    # Keystone correctness test for the excited-only dispatch key.
    # Under LINEAR the (N+1) embedding has psi[0]=1 deterministically
    # (|g> decoupled from H and every L), and the noise is keyed per
    # mode-index, so with the same SEED, TLEN, TAU, MAXHIER, and
    # matched bath modes per site the excited-block dynamics are
    # bitwise identical between the two layouts. Their C(t) outputs
    # must therefore agree at machine precision. A regression in the
    # excited-only readout (wrong seed normalization, wrong conj convention,
    # off-by-one in the time slice) breaks this immediately.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])

    traj_embed = _make_traj(2, seed=7, eom='LINEAR')
    C1_embed = calc_absorption_response(traj_embed, list_transition_dipoles, E_1, 20.0, 2.0)

    traj_excited = _make_traj_excited_only(2, seed=7, eom='LINEAR')
    C1_excited = calc_absorption_response(
        traj_excited, list_transition_dipoles, E_1, 20.0, 2.0
    )

    np.testing.assert_allclose(C1_embed, C1_excited, atol=1e-10)


# ------------------------------------------------------------
# TEST: rejects NONLINEAR with N-dim trajectory
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_absorption_response_rejects_nonlinear_excited_only():
    # The N-dim key is LINEAR-only because the NONLINEAR mean-field
    # term <L^dagger> = <psi|L^dagger|psi> / <psi|psi> requires the
    # |g> bra-norm denominator from the embedded layout. Combining
    # NONLINEAR with an excited-only trajectory must raise.
    traj = _make_traj_excited_only(2, eom='NONLINEAR')
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(UnsupportedRequest, match='vector_excited_only_NL'):
        calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)


# ------------------------------------------------------------
# TEST: rejects trajectory dim that is neither n_site nor n_site+1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_absorption_response_rejects_bad_dim():
    # Trajectory NSTATES=2 with list_transition_dipoles of n_site=4 means accepted
    # dims are {4, 5}; 2 is in neither bucket. Confirm the dispatcher
    # raises rather than silently picking a key.
    traj = _make_traj_excited_only(2, eom='LINEAR')  # NSTATES = 2
    list_transition_dipoles = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])  # n_site = 4, accepted dims = {4, 5}
    E_1 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='trajectory dim'):
        calc_absorption_response(traj, list_transition_dipoles, E_1, 4.0, 2.0)


# ------------------------------------------------------------
# TEST: tensor and vector excited-only LINEAR agree
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_tensor_absorption_excited_only_matches_vector():
    # The N-dim key runs through HopsTensorTrajectory's tensor EOM
    # without the |g> leg the existing tensor tests exercise. Confirm
    # tensor and vector backends produce the same C(t) for the
    # excited-only LINEAR layout, mirroring test_tensor_absorption_
    # matches_vector for the embedded key.
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])

    traj_vec = _make_traj_excited_only(2, seed=7, eom='LINEAR')
    C1_vec = calc_absorption_response(traj_vec, list_transition_dipoles, E_1, 10.0, 2.0)

    traj_tensor = _make_tensor_traj_excited_only(2, seed=7, eom='LINEAR')
    C1_tensor = calc_absorption_response(
        traj_tensor, list_transition_dipoles, E_1, 10.0, 2.0
    )

    np.testing.assert_allclose(C1_tensor, C1_vec, atol=1e-6)


# Shared builder for the vacuum-vs-gs-core regression tests below.
# The leading underscore keeps pytest from collecting it as a test.
def _make_tensor_traj_for_compare(n_site, use_gs, seed=0, eom='NONLINEAR',
                                  noise_model='FFT_FILTER'):
    """
    Tensor trajectory in either GS-as-state-core or vacuum convention,
    with caller-selected EOM and noise model so the absorption (LINEAR
    + ZERO) and fluorescence (NL + FFT_FILTER) regressions can share a
    single builder.

    use_gs=True : NSTATES = n_site + 1, H placed at [1:, 1:], L-ops at
                  site+1, method='fullstate'.  The GS slot
                  rides as the index-0 physical state.
    use_gs=False: NSTATES = n_site, H placed at the bare excited block,
                  L-ops at site, method='number' with
                  flag_gs_vacuum=True.  Same physics, no GS slot.
    """
    # Site energies: first site elevated, rest at 0.  Generalizes the
    # earlier [100.0, 0.0][:n_site] pattern to arbitrary n_site so the
    # trimer regression below can use the same helper.
    list_site_energies = np.array(
        [100.0] + [0.0] * (n_site - 1), dtype=np.complex128,
    )

    if use_gs:
        n_state = n_site + 1
        H2_sys = np.zeros((n_state, n_state), dtype=np.complex128)
        H2_exc = (
            np.diag(list_site_energies)
            + np.diag(np.full(n_site - 1, -50.0, dtype=np.complex128), k=1)
            + np.diag(np.full(n_site - 1, -50.0, dtype=np.complex128), k=-1)
        )
        H2_sys[1:, 1:] = H2_exc

        list_lop = []
        for i in range(n_site):
            lop = np.zeros((n_state, n_state), dtype=np.float64)
            lop[i + 1, i + 1] = 1.0
            list_lop.append(lop)
    else:
        n_state = n_site
        H2_sys = (
            np.diag(list_site_energies)
            + np.diag(np.full(n_site - 1, -50.0, dtype=np.complex128), k=1)
            + np.diag(np.full(n_site - 1, -50.0, dtype=np.complex128), k=-1)
        )

        list_lop = []
        for i in range(n_site):
            lop = np.zeros((n_state, n_state), dtype=np.float64)
            lop[i, i] = 1.0
            list_lop.append(lop)

    gw_sysbath = [[10.0, 10.0]] * n_site

    dict_sys_param = {
        'HAMILTONIAN': H2_sys,
        'GW_SYSBATH': gw_sysbath,
        'L_HIER': list_lop,
        'L_NOISE1': list_lop,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_sysbath,
    }
    dict_noise_param = {
        'SEED': seed,
        'MODEL': noise_model,
        'TLEN': 200.0,
        'TAU': 1.0,
    }
    dict_hier_param = {'MAXHIER': 2}
    dict_eom_param = {'EQUATION_OF_MOTION': eom}
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': (
            'fullstate' if use_gs
            else 'number'
        ),
        'BOND_DIM_MAX': 20,
    }

    return HopsTensorTrajectory(
        dict_sys_param,
        noise_param=dict_noise_param,
        hierarchy_param=dict_hier_param,
        eom_param=dict_eom_param,
        tensor_param=tensor_param,
    )


# ------------------------------------------------------------
# TEST: absorption — vacuum convention matches gs_core dimer
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_absorption_vacuum_matches_gs_core_dimer():
    """
    LINEAR + ZERO-noise absorption on a dimer.  The vacuum-convention
    trajectory (NSTATES=2, all-zeros |g>) and the gs_core trajectory
    (NSTATES=3, psi[0]=1 |g>) represent the same physical system and
    must produce the same C_abs(t) within MPS truncation noise.

    Uses asymmetric non-orthogonal dipoles so both site W-blocks of
    the raise MPO contribute non-trivially: orthogonal mu = [1, 0]
    would zero the mu_2 column and silently hide site-2 bugs.
    """
    n_site = 2
    list_transition_dipoles = np.array([[0.6, 0.0, 0.3], [0.4, 0.5, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    t_max = 10.0
    t_step = 2.0

    traj_gs = _make_tensor_traj_for_compare(
        n_site, use_gs=True, seed=11, eom='LINEAR', noise_model='ZERO',
    )
    C_gs = calc_absorption_response(traj_gs, list_transition_dipoles, E_1, t_max, t_step)

    traj_vac = _make_tensor_traj_for_compare(
        n_site, use_gs=False, seed=11, eom='LINEAR', noise_model='ZERO',
    )
    C_vac = calc_absorption_response(
        traj_vac, list_transition_dipoles, E_1, t_max, t_step,
    )

    assert C_gs.shape == C_vac.shape
    np.testing.assert_allclose(C_vac, C_gs, atol=1e-12, rtol=1e-10)


# ------------------------------------------------------------
# TEST: absorption NL — vacuum convention matches gs_core dimer
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_absorption_vacuum_matches_gs_core_dimer_nl():
    """
    NL absorption on a dimer with a fixed-seed FFT_FILTER noise
    realization.  Same noise -> same trajectory dynamics in both
    conventions; C_abs(t) must agree within MPS truncation noise.

    Exercises the flag_gs_vacuum=True branch of
    HopsTensorEOM.compute_z_mem_update (adds |gs_amp|^2 to the
    <L> mean-field denominator under NL), which the LINEAR + ZERO
    sibling above does not touch.
    """
    n_site = 2
    list_transition_dipoles = np.array([[0.6, 0.0, 0.3], [0.4, 0.5, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    t_max = 10.0
    t_step = 2.0

    traj_gs = _make_tensor_traj_for_compare(n_site, use_gs=True, seed=17)
    C_gs = calc_absorption_response(traj_gs, list_transition_dipoles, E_1, t_max, t_step)

    traj_vac = _make_tensor_traj_for_compare(n_site, use_gs=False, seed=17)
    C_vac = calc_absorption_response(
        traj_vac, list_transition_dipoles, E_1, t_max, t_step,
    )

    assert C_gs.shape == C_vac.shape
    np.testing.assert_allclose(C_vac, C_gs, atol=1e-12, rtol=1e-10)


# ------------------------------------------------------------
# TEST: fluorescence — vacuum convention matches gs_core dimer
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fluorescence_vacuum_matches_gs_core_dimer():
    """
    NL fluorescence on a dimer with a fixed-seed FFT_FILTER noise
    realization.  Same noise -> same trajectory dynamics in both
    conventions; C_fl(t) must agree within MPS truncation noise.
    """
    n_site = 2
    list_transition_dipoles = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([1.0, 0.0, 0.0])
    t2 = 4.0
    t3_max = 10.0
    t_step = 2.0

    traj_gs = _make_tensor_traj_for_compare(n_site, use_gs=True, seed=13)
    C_gs = calc_fluorescence_response(
        traj_gs, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step,
    )

    traj_vac = _make_tensor_traj_for_compare(n_site, use_gs=False, seed=13)
    C_vac = calc_fluorescence_response(
        traj_vac, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step,
    )

    assert C_gs.shape == C_vac.shape
    np.testing.assert_allclose(C_vac, C_gs, atol=1e-12, rtol=1e-10)


# ------------------------------------------------------------
# TEST: fluorescence on a trimer with non-uniform dipoles.
# Exercises the interior state-core W-matrices of both new
# MPOs (bond-dim-2 raise and bond-dim-3 lower+ident) — N=2
# has no interior core, so this case is the first one that
# actually contracts a (3, 2, 2, 3) interior block.
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_fluorescence_vacuum_matches_gs_core_trimer():
    """
    Trimer fluorescence regression with multi-cartesian dipoles and
    distinct E_1 != E_sig, so that:
      - y / z columns of list_transition_dipoles contribute (not just x),
      - the raise pathway (mu . E_1) and the lower pathway
        (mu . E_sig) use distinct effective dipole vectors.
    Catches bugs in non-x cartesian channels and in any asymmetry
    between the raise and lower paths.
    """
    n_site = 3
    list_transition_dipoles = np.array([
        [1.0, 0.2, 0.3],
        [0.7, 0.5, 0.1],
        [0.4, 0.6, 0.2],
    ])
    E_1 = np.array([1.0, 0.0, 0.0])
    E_sig = np.array([0.0, 1.0, 0.0])
    t2 = 4.0
    t3_max = 10.0
    t_step = 2.0

    traj_gs = _make_tensor_traj_for_compare(n_site, use_gs=True, seed=29)
    C_gs = calc_fluorescence_response(
        traj_gs, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step,
    )

    traj_vac = _make_tensor_traj_for_compare(n_site, use_gs=False, seed=29)
    C_vac = calc_fluorescence_response(
        traj_vac, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step,
    )

    assert C_gs.shape == C_vac.shape
    np.testing.assert_allclose(C_vac, C_gs, atol=1e-12, rtol=1e-10)


# ============================================================
# TEST SUITE: _spectroscopy_key()
# ============================================================

# ------------------------------------------------------------
# TEST: vector trajectory at dim n_site+1 maps to the embedded key
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_vector_embedded():
    traj = _make_traj(2)
    assert _spectroscopy_key(traj, 2) == SpectroscopyDispatch(
        'vector', None, 'embedded', 'NL'
    )


# ------------------------------------------------------------
# TEST: vector trajectory at dim n_site maps to the excited-only key
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_vector_excited_only():
    traj = _make_traj_excited_only(2)
    assert _spectroscopy_key(traj, 2) == SpectroscopyDispatch(
        'vector', None, 'excited_only', 'LINEAR'
    )


# ------------------------------------------------------------
# TEST: fullstate tensor trajectory maps to the embedded key
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_fullstate_embedded():
    traj = _make_tensor_traj(2)
    assert _spectroscopy_key(traj, 2) == SpectroscopyDispatch(
        'tensor', 'fullstate', 'embedded', 'NL'
    )


# ------------------------------------------------------------
# TEST: number trajectory takes the vacuum key and flags the wavefunction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_number_vacuum_sets_flag():
    traj = _make_tensor_traj_sn(2)
    assert _spectroscopy_key(traj, 2) == SpectroscopyDispatch(
        'tensor', 'number', 'vacuum', 'NL'
    )
    assert traj.wavefunction.flag_gs_vacuum is True


# ------------------------------------------------------------
# TEST: number representation rejects the embedded convention
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_number_embedded_raises():
    # Dim 2 read as the embedded layout of a 1-site system; number trajectories
    # only support the vacuum convention.
    traj = _make_tensor_traj_sn(2)
    with pytest.raises(UnsupportedRequest, match='tensor_number_embedded'):
        _spectroscopy_key(traj, 1)


# ------------------------------------------------------------
# TEST: rejects an EOM outside the supported set
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_rejects_unsupported_eom():
    traj = _make_traj(2, eom='NORMALIZED NONLINEAR')
    with pytest.raises(UnsupportedRequest, match='NORMALIZED NONLINEAR'):
        _spectroscopy_key(traj, 2)


# ------------------------------------------------------------
# TEST: rejects a Hilbert dimension that is neither n_site nor n_site+1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_rejects_bad_dimension():
    traj = _make_traj(2)
    with pytest.raises(ValueError, match='!='):
        _spectroscopy_key(traj, 5)


# ------------------------------------------------------------
# TEST: rejects an already-initialized trajectory
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_spectroscopy_key_rejects_initialized_traj():
    traj = _make_traj(2)
    P1_psi_0 = np.zeros(3, dtype=np.complex128)
    P1_psi_0[0] = 1.0
    traj.initialize(P1_psi_0)
    with pytest.raises(ValueError, match='initialized trajectory'):
        _spectroscopy_key(traj, 2)
