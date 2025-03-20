import numpy as np
from mesohops.util.physical_constants import kB, hbar

__title__ = "bath_corr_functions"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"

# Bath Correlation Functions
# --------------------------
# Note: Remember that the function arguments determine the
# parameters that are in the 'GW_SYSBATH' slot in the system
# dictionary.


def bcf_exp(t_axis, g, w):
    """
    This is the form of the correlation function:
    alpha(t) = \displastyle\ g exp(-w*t)

    Parameters
    ----------
    1. t_axis : np.array(float)
                Time axis [units: fs].

    2. g : complex
           Exponential prefactor [units: cm^-2].

    3. w : complex
           Exponent [units: cm^-1].

    Returns
    -------
    1. bcf : np.array(complex)
             Exponential bath correlation function sampled on t_axis.
    """
    return g * np.exp(-w * t_axis / hbar)


def bcf_convert_sdl_to_exp(lambda_sdl, gamma_sdl, omega_sdl, temp):
    """
    Converts a shifted drude-lorentz spectral density parameters to the exponential
    equivalent.

    NOTE: THIS WILL NEED TO BE REPLACED WITH A MORE ROBUST FITTING
          ROUTINE SIMILAR TO WHAT EISFELD HAS DONE PREVIOUSLY.

    Parameters
    ----------
    1. lambda_sdl : float
                    Reorganization energy [units: cm^-1].

    2. gamma_sdl : float
                   Reorganization time scale [units: cm^-1].

    3. omega_sdl : float
                   Vibrational frequency [units: cm^-1].

    4. temp : float
              Temperature [units: K].

    Returns
    -------
    1. g_exp : complex
               Exponential prefactor [units: cm^-2].

    2. w_exp : complex
               Exponent [units: cm^-1].
    """
    beta = 1 / (kB * temp)
    g_exp = 2 * lambda_sdl / beta - 1j * lambda_sdl * gamma_sdl
    w_exp = gamma_sdl - 1j * omega_sdl

    return (g_exp, w_exp)

def bcf_convert_dl_ud_to_exp(lambda_dl, gamma_dl, omega_dl, temp):
    """
    Converts underdamped Drude-Lorentz spectral density parameters to the exponential
    equivalent. Assumes that omega_dl (the underdamped frequency) is larger than
    gamma_dl (the reorganization timescale). Does not account for Matsubara modes.

    Parameters
    ----------
    1. lambda_sdl : float
                    Reorganization energy [units: cm^-1].

    2. gamma_sdl : float
                   Reorganization time scale [units: cm^-1].

    3. omega_sdl : float
                   Vibrational frequency [units: cm^-1].

    4. temp : float
              Temperature [units: K].

    Returns
    -------
    1. list_modes: list(complex)
                   List of the exponential modes that comprise the correlation
                   function, alternating gs and ws (complex, [cm^-2] and [cm^-1],
                   representing the constant prefactor and exponential decay rate,
                   respectively)
    """
    beta = 1 / (kB * temp)
    xi = np.sqrt(omega_dl**2 - gamma_dl**2)
    prefactor_base = (lambda_dl * omega_dl**2)/(2 * xi)
    w_1 = xi + 1j*gamma_dl
    w_2 = -1*xi + 1j*gamma_dl
    g_1 = prefactor_base
    g_2 = -prefactor_base
    coth_w_1 = 1 / np.tanh(w_1*beta/2)
    coth_w_2 = 1 / np.tanh(w_2*beta/2)
    g_1 += (coth_w_1 - np.conj(coth_w_2)) * prefactor_base
    g_2 += (-coth_w_2 + np.conj(coth_w_1)) * prefactor_base

    # # Test to prove that this expression is the same as the high-temperature
    # # approximation of equation S52 from Bennet et al., Supplementary Information:
    # # Mechanistic regimes of vibronic transport in a heterodimer and the design
    # # principle of incoherent vibronic transport in phycobiliproteins, J. Phys. Chem.
    # # Lett., 2018, https://doi.org/10.1021/acs.jpclett.8b00844:
    # t_axis = np.arange(0, 0.21, 0.01)
    # exp_form = g_1*np.exp(1j*w_1*t_axis) + g_2*np.exp(1j*w_2*t_axis)
    # analytic_form = (lambda_dl*(gamma_dl**2 + xi**2)/xi) * np.exp(-1*gamma_dl*t_axis)\
    #                 * (2*(np.sin(beta*gamma_dl)*np.sin(xi*t_axis) + np.sinh(
    #     beta*xi)*np.cos(xi*t_axis))/(np.cosh(beta*xi)-np.cos(beta*gamma_dl)) +
    #                    1j*np.sin(xi*t_axis))
    return [g_1, -1j*w_1, g_2, -1j*w_2]


def bcf_convert_dl_to_exp_with_Matsubara(lambda_dl, gamma_dl, temp, k_matsubara):
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
