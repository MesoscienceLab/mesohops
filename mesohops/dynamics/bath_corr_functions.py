import numpy as np
from pyhops.util.physical_constants import kB, hbar

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
    This is the form of the correlation function
    alpha(t) = \displastyle\ g exp(-w*t)

    PARAMETERS
    ----------
    1. t_axis : array
                array of time points
    2. g : complex floating point number
           the exponential prefactor [cm^-2]
    3. w : complex floating point number
           the exponent [cm^-1]

    RETURNS
    -------
    1. bcf : array
             the exponential bath correlation function sampled on t_axis
    """
    return g * np.exp(-w * t_axis / hbar)


def bcf_convert_sdl_to_exp(lambda_sdl, gamma_sdl, omega_sdl, temp):
    """
    This function converts a shifted drude-lorentz spectral density
    parameters to the exponential equivalent.

    NOTE: THIS WILL NEED TO BE REPLACED WITH A MORE ROBUST FITTING
          ROUTINE SIMILAR TO WHAT EISFELD HAS DONE PREVIOUSLY.

    PARAMETERS
    ----------
    1. lambda_sdl : float
                    the reorganization energy [cm^-1]
    2. gamma_sdl : float
                   the reorganization time scale [cm^-1]
    3. omega_sdl : float
                   the vibrational frequency [cm^-1]
    4. temp : float
              the temperature [K]

    RETURNS
    -------
    1. g_exp : complex
               the exponential prefactor [cm^-2]
    2. w_exp : complex
               the exponent [cm^-1]
    """
    beta = 1 / (kB * temp)
    g_exp = 2 * lambda_sdl / beta - 1j * lambda_sdl * gamma_sdl
    w_exp = gamma_sdl - 1j * omega_sdl

    return (g_exp, w_exp)

def bcf_convert_dl_to_exp_with_Matsubara(lambda_dl, gamma_dl, temp, k_matsubara):
    """
    This function gives the high temperature mode from the Drude-Lorentz spectral
    density with a user-selected number of Matsubara frequencies and the
    corresponding corrections to the high-temperature mode.

    PARAMETERS
    ----------
    1. lambda_dl : float
                    the reorganization energy [cm^-1]
    2. gamma_dl : float
                   the reorganization time scale [cm^-1]
    3. temp : float
              the temperature [K]
    4. k_matsubara : int
                     the number of Matsubara frequencies [number]

    RETURNS
    -------
    1. list_modes: list
                   A list of the exponential modes that comprise the correlation
                   function, alternating gs and ws (complex, [cm^-2] and [cm^-1],
                   representing the constant prefactor and exponential decay rate,
                   respectively)

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