import numpy as np
from mesohops.util.physical_constants import kB, hbar

__title__ = "bath_corr_functions"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"

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
    1. lamnda_sdl : float
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
