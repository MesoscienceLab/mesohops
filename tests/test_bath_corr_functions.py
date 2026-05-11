from mesohops.util.bath_corr_functions import *
from mesohops.util.physical_constants import kB, hbar
import numpy as np
from scipy import integrate
import pytest


def test_bcf_convert_dl_to_exp():
    """
    Tests that bcf_convert_dl_to_exp returns the correct list of
    exponentials in the correct format, and that it approaches the analytic limit of
    the hyperbolic cotangent real portion of the low-temperature mode.
    """
    e_lambda = 100
    gamma = 20
    temp = 300
    mats_const = np.pi*2*kB*temp
    # Tests that the 0-Matsubara mode limit reproduces the pure high temperature
    # approximation
    overdamped_analytic = [complex(2 * e_lambda * temp * kB - 1j * e_lambda * gamma),
                           complex(gamma)]
    assert (bcf_convert_dl_to_exp(e_lambda, gamma, temp, 0) ==
            overdamped_analytic)
    # Tests that the function returns a list with 2 entries per mode
    kmats_10000 = bcf_convert_dl_to_exp(e_lambda, gamma, temp, 10000)
    assert len(kmats_10000) == 20002
    # Test that Matsubara modes are correct
    for n in range(3,len(kmats_10000),2):
        assert kmats_10000[n] == mats_const*(n-1)/2
    # Test that 10000 Matsubara modes properly returns the correct hyperbolic
    # cotangent real portion of the low-temperature mode's constant prefactor
    np.testing.assert_allclose(np.real(kmats_10000[0]), e_lambda*gamma/np.tan(
        gamma/kB/temp/2))


# ============================================================
# TEST SUITE: bcf_convert_bo_to_exp()
# ============================================================

def _bcf_from_modes(list_modes, t_axis):
    """Reconstruct BCF from exponential mode list over a time axis in [fs]."""

    C1_bcf = np.zeros(len(t_axis), dtype=complex)
    for i in range(0, len(list_modes), 2):
        g = list_modes[i]
        w = list_modes[i + 1]
        C1_bcf += g * np.exp(-w * t_axis / hbar)
    return C1_bcf


# ------------------------------------------------------------
# TEST: mode count
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_underdamped_mode_count():
    """
    Test
    ----
    Tests that bcf_convert_bo_to_exp returns the correct number of modes.

    Case
    ----
    0 Matsubara modes returns 4 entries (2 BO poles x 2 entries each).
    3 Matsubara modes returns 10 entries (2 BO + 3 Matsubara, 2 entries each).
    """
    # 0 Matsubara: 2 BO poles only
    modes_0 = bcf_convert_bo_to_exp(50, 10, 200, 300, k_matsubara=0)
    assert len(modes_0) == 4

    # 3 Matsubara: 2 BO + 3 thermal
    modes_3 = bcf_convert_bo_to_exp(50, 10, 200, 300, k_matsubara=3)
    assert len(modes_3) == 10


# ------------------------------------------------------------
# TEST: critical damping raises ValueError
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_critical_damping_raises():
    """
    Test
    ----
    Tests that critical damping (gamma = omega) raises a ValueError.

    Case
    ----
    gamma = omega = 100 should raise ValueError because poles are degenerate.
    """
    with pytest.raises(ValueError):
        bcf_convert_bo_to_exp(50, 100, 100, 300)


# ------------------------------------------------------------
# TEST: underdamped pole structure
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_underdamped_pole_structure():
    """
    Test
    ----
    Tests that the BO pole decay rates have the correct analytic structure
    in the underdamped regime and that each g is correctly paired with its w.

    Case
    ----
    gamma=10, omega=200 => omega_d = sqrt(200^2 - 10^2). The two decay rates
    w = -i*pole should have Re(w) = gamma and Im(w) = +/- omega_d. The first
    mode corresponds to the omega_plus pole, the second to omega_minus.
    """
    lambda_bo = 50
    gamma_bo = 10
    omega_bo = 200
    temp = 300
    beta = 1 / (kB * temp)
    modes = bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                                   k_matsubara=0)
    g_plus = modes[0]
    w_plus = modes[1]
    g_minus = modes[2]
    w_minus = modes[3]
    omega_d = np.sqrt(omega_bo**2 - gamma_bo**2)

    # Decay rates: Re(w) = gamma, Im(w) = +/- omega_d
    np.testing.assert_allclose(w_plus.real, gamma_bo, rtol=1e-10)
    np.testing.assert_allclose(w_minus.real, gamma_bo, rtol=1e-10)
    np.testing.assert_allclose(w_plus.imag, -omega_d, rtol=1e-10)
    np.testing.assert_allclose(w_minus.imag, omega_d, rtol=1e-10)

    # Verify g/w pairing: compute expected g at each pole independently
    omega_plus = omega_d + 1j * gamma_bo
    omega_minus = -omega_d + 1j * gamma_bo
    for pole, g_actual in [(omega_plus, g_plus), (omega_minus, g_minus)]:
        denom = -(omega_bo**2 - pole**2) + 2 * gamma_bo**2
        coth_val = 1 / np.tanh(beta * pole / 2)
        g_expected = (1j * lambda_bo * gamma_bo * omega_bo**2
                      * (coth_val - 1) / denom)
        np.testing.assert_allclose(g_actual, g_expected, rtol=1e-10)


# ------------------------------------------------------------
# TEST: overdamped pole structure
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_overdamped_pole_structure():
    """
    Test
    ----
    Tests that the BO pole decay rates are purely real in the overdamped
    regime and that each g is correctly paired with its w.

    Case
    ----
    gamma=300, omega=50 => kappa = sqrt(300^2 - 50^2). Decay rates should be
    purely real: gamma +/- kappa. The first mode corresponds to the omega_plus
    pole (rate gamma + kappa), the second to omega_minus (rate gamma - kappa).
    """
    lambda_bo = 100
    gamma_bo = 300
    omega_bo = 50
    temp = 300
    beta = 1 / (kB * temp)
    modes = bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                                   k_matsubara=0)
    g_plus = modes[0]
    w_plus = modes[1]
    g_minus = modes[2]
    w_minus = modes[3]
    kappa = np.sqrt(gamma_bo**2 - omega_bo**2)

    # Purely real decay rates
    np.testing.assert_allclose(w_plus.imag, 0, atol=1e-10)
    np.testing.assert_allclose(w_minus.imag, 0, atol=1e-10)

    # omega_plus pole -> w = gamma + kappa, omega_minus -> w = gamma - kappa
    np.testing.assert_allclose(w_plus.real, gamma_bo + kappa, rtol=1e-10)
    np.testing.assert_allclose(w_minus.real, gamma_bo - kappa, rtol=1e-10)

    # Verify g/w pairing: compute expected g at each pole independently
    omega_d = np.sqrt(omega_bo**2 - gamma_bo**2 + 0j)
    omega_plus_pole = omega_d + 1j * gamma_bo
    omega_minus_pole = -omega_d + 1j * gamma_bo
    for pole, g_actual in [(omega_plus_pole, g_plus),
                           (omega_minus_pole, g_minus)]:
        denom = -(omega_bo**2 - pole**2) + 2 * gamma_bo**2
        coth_val = 1 / np.tanh(beta * pole / 2)
        g_expected = (1j * lambda_bo * gamma_bo * omega_bo**2
                      * (coth_val - 1) / denom)
        np.testing.assert_allclose(g_actual, g_expected, rtol=1e-10)


# ------------------------------------------------------------
# TEST: Matsubara decay rates
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_matsubara_decay_rates():
    """
    Test
    ----
    Tests that Matsubara mode decay rates equal nu_k = 2*pi*k / beta.

    Case
    ----
    T=300 K, 5 Matsubara modes. Decay rates should be k * 2*pi*kB*T.
    """
    temp = 300
    beta = 1 / (kB * temp)
    modes = bcf_convert_bo_to_exp(50, 10, 200, temp, k_matsubara=5)

    for k in range(1, 6):
        w_k = modes[4 + 2 * (k - 1) + 1]  # skip 4 BO entries
        nu_k = 2 * np.pi * k / beta
        np.testing.assert_allclose(w_k, nu_k, rtol=1e-10)


# ------------------------------------------------------------
# TEST: Matsubara prefactors
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_matsubara_prefactors():
    """
    Test
    ----
    Tests that Matsubara prefactors match the analytic residue formula
    g_k = (2i/beta) * J(i*nu_k), where J is the Brownian oscillator spectral
    density evaluated at imaginary Matsubara frequencies.

    Case
    ----
    lambda=50, gamma=10, omega=200, T=300 K, 3 Matsubara modes.
    """
    lambda_bo = 50
    gamma_bo = 10
    omega_bo = 200
    temp = 300
    beta = 1 / (kB * temp)
    modes = bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                                   k_matsubara=3)

    def j_bo(w):
        """Brownian oscillator spectral density."""
        return (4 * lambda_bo * gamma_bo * omega_bo**2 * w
                / ((omega_bo**2 - w**2)**2 + 4 * gamma_bo**2 * w**2))

    for k in range(1, 4):
        g_k = modes[4 + 2 * (k - 1)]
        nu_k = 2 * np.pi * k / beta
        # Residue formula: g_k = (2i / beta) * J(i * nu_k)
        g_expected = 2j / beta * j_bo(1j * nu_k)
        np.testing.assert_allclose(g_k, g_expected, rtol=1e-10)


# ------------------------------------------------------------
# TEST: C(0) self-consistency
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_t0_value():
    """
    Test
    ----
    Tests that C(0) from the exponential decomposition agrees with the
    zero-Matsubara analytic value at t=0.

    Case
    ----
    At t=0, each mode contributes just g. The sum of all g values should give
    the correct C(0). We check self-consistency: sum of prefactors at t=0 from
    the function equals manually computed residue prefactors at t=0.
    """
    lambda_bo = 50
    gamma_bo = 10
    omega_bo = 200
    temp = 300
    beta = 1 / (kB * temp)

    modes = bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                                   k_matsubara=0)

    # C(0) from modes
    c0_modes = sum(modes[i] for i in range(0, len(modes), 2))

    # C(0) from residue formula directly
    omega_d = np.sqrt(omega_bo**2 - gamma_bo**2 + 0j)
    omega_plus = omega_d + 1j * gamma_bo
    omega_minus = -omega_d + 1j * gamma_bo
    c0_analytic = 0
    for pole in [omega_plus, omega_minus]:
        denom = -(omega_bo**2 - pole**2) + 2 * gamma_bo**2
        coth_val = 1 / np.tanh(beta * pole / 2)
        c0_analytic += 1j * lambda_bo * gamma_bo * omega_bo**2 * (coth_val - 1) / denom

    np.testing.assert_allclose(c0_modes, c0_analytic, rtol=1e-10)


# ------------------------------------------------------------
# TEST: low temperature
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_low_temperature():
    """
    Test
    ----
    Tests that the function produces finite results at low temperature where
    beta is large and coth can overflow.

    Case
    ----
    T=1 K with underdamped parameters. All prefactors and decay rates should
    be finite (no NaN or Inf).
    """
    modes = bcf_convert_bo_to_exp(50, 10, 200, 1.0, k_matsubara=5)
    for val in modes:
        assert np.isfinite(val), f'Non-finite value in modes: {val}'


# ------------------------------------------------------------
# TEST: near-critical damping stability
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_near_critical():
    """
    Test
    ----
    Tests that parameters very close to the critical damping boundary still
    produce finite results and emit a warning.

    Case
    ----
    gamma = 100, omega = 100 + 1e-6. Should not raise, should warn about
    near-critical damping, and all mode values should be finite.
    """
    with pytest.warns(UserWarning, match='Near-critical damping'):
        modes = bcf_convert_bo_to_exp(50, 100, 100 + 1e-6, 300, k_matsubara=0)
    for val in modes:
        assert np.isfinite(val), f'Non-finite value near critical: {val}'


# ------------------------------------------------------------
# TEST: strongly overdamped limit recovers Drude-Lorentz
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_overdamped_limit_matches_dl():
    """
    Test
    ----
    Tests that the BO function recovers the Drude-Lorentz BCF in the strongly
    overdamped limit (gamma >> omega).

    Case
    ----
    lambda=100, gamma=5000, omega=100, T=300 K. In this limit the effective
    DL rate is gamma_D ~ omega^2 / (2*gamma). Compares reconstructed BCFs
    over a time axis long enough to see the decay.
    """
    lambda_bo = 100
    gamma_bo = 5000
    omega_bo = 100
    temp = 300
    gamma_dl_eff = omega_bo**2 / (2 * gamma_bo)

    # Time axis in [fs]; gamma_D ~ 1 cm^-1, so decay time ~ hbar/gamma_D
    t_axis = np.linspace(10, 3000, 50)

    bo_modes = bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                                     k_matsubara=10)
    dl_modes = bcf_convert_dl_to_exp(lambda_bo, gamma_dl_eff, temp,
                                     k_matsubara=10)

    C1_bcf_bo = _bcf_from_modes(bo_modes, t_axis)
    C1_bcf_dl = _bcf_from_modes(dl_modes, t_axis)

    np.testing.assert_allclose(C1_bcf_bo.real, C1_bcf_dl.real, rtol=5e-2)
    np.testing.assert_allclose(C1_bcf_bo.imag, C1_bcf_dl.imag, rtol=5e-2)


# ------------------------------------------------------------
# TEST: Matsubara denominator near zero
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_matsubara_denom_near_zero():
    """
    Test
    ----
    Tests behavior when a Matsubara frequency nearly coincides with a spectral
    density pole, making denom_mats close to zero.

    Case
    ----
    The Matsubara denominator is (omega^2 + nu_k^2)^2 - 4*gamma^2*nu_k^2.
    This vanishes when (omega^2 + nu_k^2) = 2*gamma*nu_k, i.e., when a
    Matsubara frequency coincides with a spectral density pole.
    Case 1: parameters that make denom_mats small for k=1.
    Case 2: large gamma that makes denom_mats small for k=3 (not k=1),
    verifying the check catches higher Matsubara modes.
    """
    # Case 1: near-zero denominator at k=1
    # nu_1 = 2*pi*kB*T. Choose gamma so that (omega^2 + nu_1^2) ~ 2*gamma*nu_1
    temp = 300
    nu_1 = 2 * np.pi * kB * temp
    omega_bo = 50
    gamma_bo = (omega_bo**2 + nu_1**2) / (2 * nu_1)

    with pytest.warns(UserWarning, match='near-zero denominator'):
        modes = bcf_convert_bo_to_exp(100, gamma_bo, omega_bo, temp,
                                       k_matsubara=1)
    for val in modes:
        assert np.isfinite(val), (
            f'Non-finite value with near-zero Matsubara denom: {val}'
        )

    # Case 2: near-zero denominator at k=3 (large gamma)
    # nu_3 = 3*2*pi*kB*T. Choose gamma so the pole coincides with k=3.
    nu_3 = 3 * 2 * np.pi * kB * temp
    gamma_bo_k3 = (omega_bo**2 + nu_3**2) / (2 * nu_3)

    with pytest.warns(UserWarning, match='k=3'):
        modes_k3 = bcf_convert_bo_to_exp(100, gamma_bo_k3, omega_bo, temp,
                                          k_matsubara=3)
    for val in modes_k3:
        assert np.isfinite(val), (
            f'Non-finite value with near-zero Matsubara denom at k=3: {val}'
        )


# ------------------------------------------------------------
# TEST: non-positive temperature raises ValueError
# ------------------------------------------------------------
def test_bcf_convert_bo_to_exp_nonpositive_temp_raises():
    """
    Test
    ----
    Tests that non-positive temperatures raise a ValueError.

    Case
    ----
    temp=0 and temp=-10 should both raise ValueError because beta diverges
    or becomes non-physical.
    """
    with pytest.raises(ValueError, match='Temperature must be positive'):
        bcf_convert_bo_to_exp(50, 10, 200, 0)
    with pytest.raises(ValueError, match='Temperature must be positive'):
        bcf_convert_bo_to_exp(50, 10, 200, -10)


# ------------------------------------------------------------
# TEST: BCF vs numerical quadrature of spectral density
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_bcf_convert_bo_to_exp_vs_quadrature():
    """
    Test
    ----
    Tests that the exponential decomposition reproduces the BCF obtained by
    direct numerical integration of the spectral density:

        C(t) = (1/pi) * int_0^inf J(w) [coth(beta*w/2) cos(wt) - i sin(wt)] dw

    This catches systematic errors in the residue formula itself, not just in
    its transcription to code.

    Case
    ----
    Underdamped regime: lambda=50, gamma=10, omega=200, T=300 K, 10 Matsubara
    modes. Compared at several time points spanning the oscillation period.
    """
    lambda_bo = 50
    gamma_bo = 10
    omega_bo = 200
    temp = 300
    k_matsubara = 10
    beta = 1 / (kB * temp)

    def j_bo(w):
        """Brownian oscillator spectral density."""
        return (4 * lambda_bo * gamma_bo * omega_bo**2 * w
                / ((omega_bo**2 - w**2)**2 + 4 * gamma_bo**2 * w**2))

    def bcf_real_integrand(w, t_cm):
        """Real part integrand: J(w) * coth(beta*w/2) * cos(w*t)."""
        return j_bo(w) * (1 / np.tanh(beta * w / 2)) * np.cos(w * t_cm)

    def bcf_imag_integrand(w, t_cm):
        """Imaginary part integrand: -J(w) * sin(w*t)."""
        return -j_bo(w) * np.sin(w * t_cm)

    # Time points in [fs], converted to [cm^-1]^-1 via hbar
    t_fs = np.array([10, 50, 100, 200, 500])
    t_cm = t_fs / hbar

    modes = bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                                   k_matsubara=k_matsubara)
    C1_bcf_modes = _bcf_from_modes(modes, t_fs)

    # Finite upper limit: J(w) ~ 1/w^3 for large w, so contributions
    # beyond 10*omega_bo are negligible.
    w_max = 10 * omega_bo
    for i, t in enumerate(t_cm):
        re_quad, _ = integrate.quad(bcf_real_integrand, 0, w_max,
                                    args=(t,), limit=500,
                                    points=[omega_bo])
        im_quad, _ = integrate.quad(bcf_imag_integrand, 0, w_max,
                                    args=(t,), limit=500,
                                    points=[omega_bo])
        c_quad = (re_quad + 1j * im_quad) / np.pi

        np.testing.assert_allclose(C1_bcf_modes[i].real, c_quad.real,
                                   rtol=1e-3)
        np.testing.assert_allclose(C1_bcf_modes[i].imag, c_quad.imag,
                                   rtol=1e-3)
