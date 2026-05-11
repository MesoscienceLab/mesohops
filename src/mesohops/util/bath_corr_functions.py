import warnings

import numpy as np
from mesohops.util.physical_constants import kB

__title__ = "bath_corr_functions"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.6"

# Bath Correlation Functions
# --------------------------
# Note: Remember that the function arguments determine the
# parameters that are in the 'GW_SYSBATH' slot in the system
# dictionary.



def bcf_convert_bo_to_exp(lambda_bo, gamma_bo, omega_bo, temp,
                          k_matsubara=0):
    """
    Converts Brownian oscillator spectral density parameters to an exponential
    bath correlation function via contour integration over the upper half-plane.

    Handles both underdamped (gamma < omega) and overdamped (gamma > omega)
    regimes using the general residue result. Raises an error for the critically
    damped case (gamma = omega) where the spectral density poles are degenerate.

    Parameters
    ----------
    1. lambda_bo: float
                  Reorganization energy [units: cm^-1].

    2. gamma_bo: float
                 Damping rate [units: cm^-1].

    3. omega_bo: float
                 Characteristic vibrational frequency [units: cm^-1].

    4. temp: float
             Temperature [units: K].

    5. k_matsubara: int
               Number of Matsubara frequency corrections.

    Returns
    -------
    1. list_modes: list(complex)
                   Exponential modes that comprise the correlation function,
                   alternating gs and ws ([units: cm^-2] and [units: cm^-1],
                   representing the constant prefactor and exponential decay
                   rate, respectively).
    """
    if np.isclose(gamma_bo, omega_bo, rtol=1e-10):
        raise ValueError(
            'Critical damping (gamma = omega) produces degenerate poles and '
            f'is not supported. Got: gamma={gamma_bo}, omega={omega_bo}.'
        )

    if temp <= 0:
        raise ValueError(
            f'Temperature must be positive. Got: temp={temp}.'
        )

    if abs(gamma_bo - omega_bo) < 1.0:
        warnings.warn(
            f'Near-critical damping (|gamma - omega| = '
            f'{abs(gamma_bo - omega_bo):.2e} cm^-1) may cause large, '
            f'poorly converged prefactors.',
            stacklevel=2,
        )

    beta = 1 / (kB * temp)

    # Upper half-plane poles of J(w): w_+/- = i*gamma +/- sqrt(Omega^2 - gamma^2)
    # The complex square root unifies the underdamped regime (real sqrt gives
    # oscillatory poles) and overdamped regime (imaginary sqrt gives purely
    # decaying poles) without branching logic.
    omega_d = np.sqrt(omega_bo**2 - gamma_bo**2 + 0j)
    omega_plus = omega_d + 1j * gamma_bo
    omega_minus = -omega_d + 1j * gamma_bo
    # Mode ordering after HOPS conversion w = -i*pole:
    #   Underdamped: w_+ = gamma - i*omega_d, w_- = gamma + i*omega_d
    #               (conjugate pair, both decay at rate gamma)
    #   Overdamped:  w_+ = gamma + kappa,     w_- = gamma - kappa
    #               (w_+ is the fast-decaying mode, w_- is the slow mode)

    # Spectral density contribution to C(t).
    # J(w) = N(w)/D(w) has simple poles where D(w) = 0. For a simple pole at
    # w_k, the residue is N(w_k) / D'(w_k) where D' is the derivative of D.
    # Here N(w) = 4*lambda*gamma*Omega^2*w and D'(w) = -4w*(Omega^2 - w^2)
    # + 8*gamma^2*w share a common factor of 4*w, leaving the simplified
    # denominator below: -(Omega^2 - w_k^2) + 2*gamma^2.
    list_modes = []
    for pole in [omega_plus, omega_minus]:
        denom = -(omega_bo**2 - pole**2) + 2 * gamma_bo**2
        coth_val = 1 / np.tanh(beta * pole / 2)
        g = (1j * lambda_bo * gamma_bo * omega_bo**2
             * (coth_val - 1) / denom)
        # Convert contour convention e^{i*w*t} to HOPS convention e^{-w*t}
        w = -1j * pole
        list_modes.extend([g, w])

    # Matsubara poles: coth(beta*w/2) has simple poles on the imaginary axis
    # at w_k = i*nu_k where nu_k = 2*pi*k/beta. The k=0 pole is cancelled
    # because J(w) vanishes linearly at w=0. Each remaining pole contributes
    # a purely real, decaying exponential with prefactor proportional to
    # J(i*nu_k) evaluated at the imaginary Matsubara frequency.
    for k in range(1, k_matsubara + 1):
        nu_k = 2 * np.pi * k / beta
        denom_mats = ((omega_bo**2 + nu_k**2)**2
                      - 4 * gamma_bo**2 * nu_k**2)
        # Warn when denom_mats is near zero: g_mats ~ 1/denom_mats diverges.
        if abs(denom_mats) < 1e-3 * (omega_bo**2 + nu_k**2)**2:
            warnings.warn(
                f'Matsubara mode k={k} has a near-zero denominator '
                f'(nu_k={nu_k:.2f} cm^-1 is close to a spectral density '
                f'pole). The prefactor may be unreliable.',
                stacklevel=2,
            )
        g_mats = (-8 * lambda_bo * gamma_bo * omega_bo**2 * nu_k
                  / (beta * denom_mats))
        list_modes.extend([g_mats, nu_k])

    return list_modes


def bcf_convert_dl_to_exp(lambda_dl, gamma_dl, temp, k_matsubara=0):
    """
    Gives the high temperature mode from the Drude-Lorentz spectral density with a
    user-selected number of Matsubara frequencies and the corresponding corrections to
    the high-temperature mode.

    Parameters
    ----------
    1. lambda_dl : float
                    Reorganization energy [units: cm^-1].

    2. gamma_dl : float
                   Reorganization time scale [units: cm^-1].

    3. temp : float
              Temperature [units: K].

    4. k_matsubara : int
                     Number of Matsubara frequencies.

    Returns
    -------
    1. list_modes: list(complex)
                   Exponential modes that comprise the correlation function, alternating
                   gs and ws ([units: cm^-2] and [units: cm^-1], representing the
                   constant prefactor and exponential decay rate, respectively).

    """
    beta = 1 / (kB * temp)
    g_exp = 2 * lambda_dl / beta - 1j * lambda_dl * gamma_dl
    w_exp = gamma_dl
    mats_mode_const = 2*np.pi/(beta)
    def J(w):
        return 2*lambda_dl*gamma_dl*w/(w**2 + gamma_dl**2)
    list_mats_modes = []
    for k in np.arange(k_matsubara)+1:
        w_mats = k*mats_mode_const
        g_mats = 2j*J(1j*w_mats)/beta
        g_exp += 2*lambda_dl*(1/beta)*(2*gamma_dl**2)/(gamma_dl**2 - w_mats**2)
        list_mats_modes += [g_mats, w_mats]

    list_modes = [g_exp, w_exp] + list_mats_modes
    return list_modes


def ishizaki_decomposition_bcf_dl(lambda_dl, gamma_dl, temp, k_matsubara, epsilon=None):
    """
    Calculates Ishizaki decomposition of a Drude-Lorentz-like spectral
    density, as detailed in https://doi.org/10.7566/JPSJ.89.015001.

    Parameters
    ----------
    1. lambda_dl : float
                    Reorganization energy [units: cm^-1].

    2. gamma_dl : float
                  Reorganization time scale [units: cm^-1].

    3. temp : float
              Temperature [units: K].

    4. k_matsubara : int
                     Number of Matsubara frequencies.

    5. epsilon : float
                     Mathematical fudge factor allowing for Gaussian-like behavior
                     of the correlation function that must be significantly smaller
                     than gamma_dl [units: cm^-1].


    Returns
    -------
    1. list_modes: list(complex)
                   List of the exponential modes that comprise the correlation
                   function, alternating gs and ws ([units: cm^-2] and [units: cm^-1],
                   representing the constant prefactor and exponential decay rate,
                   respectively).

    """
    # Epsilon automatically set to gamma_dl/10
    if epsilon is None:
        epsilon = gamma_dl/10
    # Generate quantitites used in Ishizaki's derivation
    GAMMA_dl = gamma_dl*2
    GAMMA_plus = GAMMA_dl + 1j*epsilon
    GAMMA_minus = GAMMA_dl - 1j*epsilon
    beta = 1 / (kB * temp)
    mats_mode_const = 2*np.pi/(beta)

    # The spectral density
    def J(w):
        return 4*lambda_dl*(GAMMA_dl**3)*w/((w**2 + GAMMA_dl**2)**2)

    # Get the first mode and its imaginary component
    g_exp = 1j*lambda_dl*GAMMA_minus/(epsilon*beta)
    g_exp_im = 1j*lambda_dl*GAMMA_plus*GAMMA_minus/(2*epsilon)
    g_exp_im_conj = np.conj(g_exp_im)
    w_exp = GAMMA_plus

    # Get the Matsubara modes and their associated correction to the real portion of
    # the first mode's prefactor
    list_mats_modes = []
    for k in np.arange(k_matsubara)+1:
        w_mats = k*mats_mode_const
        g_mats = (2j/beta)*J(1j*w_mats)
        def E_tilde_k(w):
            return (2*w**2)/((w**2 - w_mats**2)*beta)
        g_exp += 2*np.real((1j*lambda_dl*GAMMA_minus/epsilon)*E_tilde_k(GAMMA_plus))
        list_mats_modes += [g_mats, w_mats]

    # Get the second mode
    g_exp_conj = np.conj(g_exp)
    w_exp_conj = np.conj(w_exp)

    return [g_exp - 1j*g_exp_im, w_exp, g_exp_conj - 1j*g_exp_im_conj, w_exp_conj] + \
           list_mats_modes
